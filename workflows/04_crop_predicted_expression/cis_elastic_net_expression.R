#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(progress)
  library(optparse)
  library(data.table)
  library(plink2R)
  library(glue)
  library(glmnet)
})

option_list <- list(
  make_option(c("-p", "--plink_prefix"), type = "character", default = NULL,
              help = "PLINK genotype file prefix", metavar = "character"),
  make_option(c("-c", "--chr"), type = "character", default = NULL,
              help = "Chromosome name without --chr_prefix", metavar = "character"),
  make_option("--chr_prefix", type = "character", default = "chr",
              help = "Chromosome name prefix used in GFF, e.g. chr or Chr", metavar = "character"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,
              help = "Output directory", metavar = "character"),
  make_option("--gff", type = "character", default = NULL,
              help = "GFF3 annotation file", metavar = "character"),
  make_option("--gene_exp", type = "character", default = NULL,
              help = "Expression matrix, genes by samples", metavar = "character"),
  make_option("--gene_exp_sep", type = "character", default = ",",
              help = "Expression matrix delimiter; use comma or \\t", metavar = "character"),
  make_option("--predict_fam", type = "character", default = NULL,
              help = "FAM file listing samples for expression prediction", metavar = "character"),
  make_option("--plink_bin", type = "character", default = Sys.which("plink"),
              help = "PLINK executable", metavar = "character"),
  make_option("--fam_id_column", type = "integer", default = 1,
              help = "FAM identifier column: 1=FID (legacy default), 2=IID", metavar = "integer"),
  make_option("--gene_id_attribute", type = "character", default = "ID",
              help = "GFF3 attribute used as the expression gene ID", metavar = "character"),
  make_option("--cis_window", type = "numeric", default = 5e4,
              help = "Cis SNP window around gene body", metavar = "numeric"),
  make_option("--train_frac", type = "numeric", default = 0.8,
              help = "Training fraction for expression-observed samples", metavar = "numeric"),
  make_option("--seed", type = "integer", default = 123,
              help = "Random seed", metavar = "integer"),
  make_option("--alpha", type = "numeric", default = 0.5,
              help = "glmnet elastic-net alpha", metavar = "numeric"),
  make_option("--cv_folds", type = "integer", default = 10,
              help = "Maximum glmnet CV folds", metavar = "integer"),
  make_option("--min_expression", type = "numeric", default = 1,
              help = "Expression threshold used by prevalence filtering", metavar = "numeric"),
  make_option("--expression_prevalence", type = "numeric", default = 0.5,
              help = "Minimum fraction of observed samples above min_expression", metavar = "numeric"),
  make_option("--max_genes", type = "integer", default = 0,
              help = "Optional debug limit; 0 means all genes", metavar = "integer")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

required_options <- c("plink_prefix", "chr", "output_dir", "gff", "gene_exp", "predict_fam")
missing_options <- required_options[vapply(required_options, function(name) {
  value <- opt[[name]]
  is.null(value) || !nzchar(as.character(value))
}, logical(1))]
if (length(missing_options) > 0) {
  stop("Missing required options: ", paste(paste0("--", missing_options), collapse = ", "), call. = FALSE)
}
if (!(opt$fam_id_column %in% c(1, 2))) {
  stop("--fam_id_column must be 1 (FID) or 2 (IID)", call. = FALSE)
}
if (!(opt$train_frac > 0 && opt$train_frac < 1)) {
  stop("--train_frac must be between 0 and 1", call. = FALSE)
}
if (opt$cv_folds < 3) {
  stop("--cv_folds must be >= 3", call. = FALSE)
}
if (!(opt$expression_prevalence >= 0 && opt$expression_prevalence <= 1)) {
  stop("--expression_prevalence must be in [0, 1]", call. = FALSE)
}

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  plink_bed = paste0(opt$plink_prefix, ".bed"),
  plink_bim = paste0(opt$plink_prefix, ".bim"),
  plink_fam = paste0(opt$plink_prefix, ".fam"),
  gff = opt$gff,
  gene_exp = opt$gene_exp,
  predict_fam = opt$predict_fam,
  plink_bin = opt$plink_bin
)

missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files: ", paste(names(missing_files), missing_files, sep = "=", collapse = "; "))
}
if (file.access(opt$plink_bin, mode = 1) != 0) {
  stop("PLINK is not executable: ", opt$plink_bin)
}

gene_exp_sep <- opt$gene_exp_sep
if (gene_exp_sep %in% c("\\t", "tab", "TAB")) {
  gene_exp_sep <- "\t"
}

safe_scc <- function(pred, obs) {
  pred <- as.numeric(pred)
  obs <- as.numeric(obs)
  ok <- is.finite(pred) & is.finite(obs)
  if (sum(ok) < 3 || sd(pred[ok]) == 0 || sd(obs[ok]) == 0) {
    return(NA_real_)
  }
  suppressWarnings(cor(pred[ok], obs[ok], method = "spearman"))
}

safe_r2 <- function(pred, obs) {
  pred <- as.numeric(pred)
  obs <- as.numeric(obs)
  ok <- is.finite(pred) & is.finite(obs)
  if (sum(ok) < 2) {
    return(NA_real_)
  }
  denom <- sum((obs[ok] - mean(obs[ok]))^2)
  if (!is.finite(denom) || denom <= 0) {
    return(NA_real_)
  }
  1 - sum((obs[ok] - pred[ok])^2) / denom
}

safe_mse <- function(pred, obs) {
  pred <- as.numeric(pred)
  obs <- as.numeric(obs)
  ok <- is.finite(pred) & is.finite(obs)
  if (sum(ok) == 0) {
    return(NA_real_)
  }
  mean((obs[ok] - pred[ok])^2)
}

impute_with_train_means <- function(x_train, x_test, x_predict) {
  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  x_predict <- as.matrix(x_predict)

  storage.mode(x_train) <- "double"
  storage.mode(x_test) <- "double"
  storage.mode(x_predict) <- "double"

  cm <- colMeans(x_train, na.rm = TRUE)
  cm[!is.finite(cm)] <- 0

  train_na <- which(is.na(x_train), arr.ind = TRUE)
  if (nrow(train_na) > 0) {
    x_train[train_na] <- cm[train_na[, 2]]
  }
  test_na <- which(is.na(x_test), arr.ind = TRUE)
  if (nrow(test_na) > 0) {
    x_test[test_na] <- cm[test_na[, 2]]
  }
  pred_na <- which(is.na(x_predict), arr.ind = TRUE)
  if (nrow(pred_na) > 0) {
    x_predict[pred_na] <- cm[pred_na[, 2]]
  }

  list(train = x_train, test = x_test, predict = x_predict)
}

select_high_expr <- function(overlap_gene_exp, cutoff, min_expression) {
  keep_count <- apply(overlap_gene_exp, 2, function(x) {
    sum(is.finite(x) & x >= min_expression)
  })
  overlap_gene_exp[, which(keep_count / nrow(overlap_gene_exp) > cutoff), drop = FALSE]
}

cleanup_temp <- function(temp_prefix) {
  temp_files <- Sys.glob(paste0(temp_prefix, ".*"))
  if (length(temp_files) > 0) {
    unlink(temp_files, force = TRUE)
  }
}

gff_data <- fread(
  cmd = paste("grep -v '^#'", shQuote(opt$gff)),
  sep = "\t",
  header = FALSE,
  stringsAsFactors = FALSE,
  fill = TRUE
)
if (ncol(gff_data) < 9) {
  stop("GFF parsing failed; expected at least 9 columns, found ", ncol(gff_data))
}
setnames(gff_data, paste0("V", seq_len(ncol(gff_data))))
gff_data <- gff_data[V3 == "gene"]
gene_ids <- sapply(gff_data$V9, function(x) {
  fields <- trimws(unlist(strsplit(x, ";", fixed = TRUE)))
  prefix <- paste0(opt$gene_id_attribute, "=")
  hit <- fields[startsWith(fields, prefix)]
  if (length(hit) == 0) NA_character_ else substring(hit[1], nchar(prefix) + 1)
})
gff_data$V9 <- unname(gene_ids)
gff_data <- gff_data[!is.na(V9) & nzchar(V9)]

chr <- opt$chr
sub <- gff_data[V1 == paste0(opt$chr_prefix, chr)]
sub <- data.frame(sub, stringsAsFactors = FALSE)

bim <- fread(paste0(opt$plink_prefix, ".bim"), sep = "\t", header = FALSE, stringsAsFactors = FALSE)
fam <- fread(paste0(opt$plink_prefix, ".fam"), sep = " ", header = FALSE, stringsAsFactors = FALSE)
fam1 <- fread(opt$predict_fam, sep = " ", header = FALSE, stringsAsFactors = FALSE)

gene_exp <- fread(opt$gene_exp, sep = gene_exp_sep, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
gene_exp <- data.frame(gene_exp, stringsAsFactors = FALSE)
rownames(gene_exp) <- gene_exp[, 1]
gene_exp <- gene_exp[, -1, drop = FALSE]

expr_samples <- colnames(gene_exp)
genotype_ids <- as.character(fam[[opt$fam_id_column]])
overlap <- intersect(genotype_ids, expr_samples)
predict_s <- as.character(fam1[[opt$fam_id_column]])
if (anyDuplicated(genotype_ids) || anyDuplicated(predict_s)) {
  stop("Selected FAM ID column contains duplicated sample identifiers", call. = FALSE)
}

if (length(overlap) < 10) {
  stop("Too few genotype/expression overlap samples: ", length(overlap))
}

gene_exp <- t(gene_exp)
overlap_gene_exp <- gene_exp[overlap, , drop = FALSE]
high_expr <- select_high_expr(
  overlap_gene_exp,
  cutoff = opt$expression_prevalence,
  min_expression = opt$min_expression
)
high_expr <- log2(high_expr + 1)

genes <- intersect(colnames(high_expr), sub$V9)
high_expr <- high_expr[, genes, drop = FALSE]
sub <- sub[match(genes, sub$V9), , drop = FALSE]

if (nrow(sub) == 0) {
  stop("No expressed genes matched chromosome ", paste0(opt$chr_prefix, chr), call. = FALSE)
}

if (opt$max_genes > 0 && nrow(sub) > opt$max_genes) {
  sub <- sub[seq_len(opt$max_genes), , drop = FALSE]
  high_expr <- high_expr[, sub$V9, drop = FALSE]
}

message(glue(
  "chr={chr}; genes={nrow(sub)}; overlap_samples={length(overlap)}; predict_samples={length(predict_s)}; output={opt$output_dir}"
))

pb <- progress_bar$new(total = nrow(sub))

train_model <- function(sub, bim, chr) {
  pred_exp <- matrix(nrow = length(predict_s), ncol = 0)
  rownames(pred_exp) <- predict_s
  metric_rows <- list()

  for (i in seq_len(nrow(sub))) {
    cur_gene <- sub[i, 9]
    left <- sub[i, 4]
    right <- sub[i, 5]
    snp_idx <- which(bim$V4 > left - opt$cis_window & bim$V4 < right + opt$cis_window)

    if (length(snp_idx) > 1) {
      snp_ids <- bim[[2]][snp_idx]
      snp_file <- file.path(opt$output_dir, paste0("chr", chr, "_gene_", i, "_topk_snp_list.txt"))
      temp_prefix <- file.path(opt$output_dir, paste0("chr", chr, "_gene_", i, "_temp_extracted"))

      write.table(snp_ids, snp_file, quote = FALSE, row.names = FALSE, col.names = FALSE)

      cmd <- glue(
        '"{opt$plink_bin}" --bfile "{opt$plink_prefix}" --extract "{snp_file}" --make-bed --out "{temp_prefix}"'
      )
      status <- system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)

      if (status == 0) {
        extracted <- read_plink(temp_prefix)
        snps <- extracted$bed

        sample_ids <- sapply(rownames(snps), function(x) {
          fields <- unlist(strsplit(x, ":", fixed = TRUE))
          if (length(fields) >= opt$fam_id_column) fields[opt$fam_id_column] else fields[1]
        })
        rownames(snps) <- unname(sample_ids)

        trainable <- intersect(overlap, rownames(snps))
        pred_samples <- intersect(predict_s, rownames(snps))

        if (length(trainable) >= 10 && length(pred_samples) > 0) {
          x <- snps[trainable, , drop = FALSE]
          y <- high_expr[trainable, cur_gene]
          names(y) <- rownames(x)
          predict_x <- snps[pred_samples, , drop = FALSE]

          if (!is.finite(sd(as.numeric(y))) || sd(as.numeric(y)) == 0) {
            cleanup_temp(temp_prefix)
            if (file.exists(snp_file)) unlink(snp_file)
            pb$tick()
            next
          }

          chr_seed <- sum(utf8ToInt(as.character(chr)))
          set.seed(opt$seed + chr_seed * 100000 + i)
          n_train <- max(3, min(nrow(x) - 2, round(opt$train_frac * nrow(x))))
          train_s <- sample(rownames(x), n_train)
          test_s <- setdiff(rownames(x), train_s)

          x_train <- x[train_s, , drop = FALSE]
          x_test <- x[test_s, , drop = FALSE]
          y_train <- y[train_s]
          y_test <- y[test_s]

          imputed <- impute_with_train_means(x_train, x_test, predict_x)

          eval_nfolds <- min(opt$cv_folds, length(y_train))
          cv_fit <- cv.glmnet(x = imputed$train, y = y_train, alpha = opt$alpha,
                              nfolds = eval_nfolds)
          lambda_optimal <- cv_fit$lambda.min
          final_fit <- glmnet(x = imputed$train, y = y_train, alpha = opt$alpha, lambda = lambda_optimal)

          test_pred <- as.numeric(predict(final_fit, newx = imputed$test))
          # After estimating held-out accuracy, refit on every expression-observed
          # sample. The prediction cohort never contributes to imputation or model fit.
          final_imputed <- impute_with_train_means(x, x[0, , drop = FALSE], predict_x)
          final_nfolds <- min(opt$cv_folds, length(y))
          final_cv <- cv.glmnet(x = final_imputed$train, y = y, alpha = opt$alpha,
                                nfolds = final_nfolds)
          final_lambda <- final_cv$lambda.min
          prediction_fit <- glmnet(x = final_imputed$train, y = y, alpha = opt$alpha,
                                   lambda = final_lambda)
          predict_expr <- as.numeric(predict(prediction_fit, newx = final_imputed$predict))

          metric_rows[[length(metric_rows) + 1]] <- data.table(
            Gene = cur_gene,
            R2 = safe_r2(test_pred, y_test),
            MSE = safe_mse(test_pred, y_test),
            SCC = safe_scc(test_pred, y_test),
            Nsnps = ncol(imputed$train),
            NTrain = length(y_train),
            NTest = length(y_test),
            LambdaEval = lambda_optimal,
            LambdaFinal = final_lambda
          )

          pred_col <- rep(NA_real_, length(predict_s))
          names(pred_col) <- predict_s
          pred_col[pred_samples] <- predict_expr
          pred_exp <- cbind(pred_exp, pred_col)
          colnames(pred_exp)[ncol(pred_exp)] <- cur_gene
        }
      }

      cleanup_temp(temp_prefix)
      if (file.exists(snp_file)) {
        unlink(snp_file)
      }
    }

    pb$tick()
  }

  metrics <- if (length(metric_rows) > 0) {
    rbindlist(metric_rows, use.names = TRUE, fill = TRUE)
  } else {
    data.table(Gene = character(), R2 = numeric(), MSE = numeric(), SCC = numeric(),
               Nsnps = integer(), NTrain = integer(), NTest = integer(),
               LambdaEval = numeric(), LambdaFinal = numeric())
  }

  list(pred_exp = pred_exp, metrics = metrics)
}

t1 <- Sys.time()
result <- train_model(sub = sub, bim = bim, chr = chr)

if (ncol(result$pred_exp) == 0) {
  pred_out <- data.frame(ID = rownames(result$pred_exp), stringsAsFactors = FALSE)
} else {
  pred_out <- data.frame(ID = rownames(result$pred_exp), result$pred_exp, stringsAsFactors = FALSE)
}

fwrite(pred_out, file.path(opt$output_dir, "Predict_Expre.tsv"),
       sep = "\t", col.names = TRUE, row.names = FALSE)

fwrite(result$metrics, file.path(opt$output_dir, "Predict_metrics.tsv"),
       sep = "\t", col.names = TRUE, row.names = FALSE)

# Backward-compatible filename; now includes R2, MSE, and SCC.
fwrite(result$metrics, file.path(opt$output_dir, "Predict_Spearman_acc.tsv"),
       sep = "\t", col.names = TRUE, row.names = FALSE)

t2 <- Sys.time()
print(t2 - t1)
