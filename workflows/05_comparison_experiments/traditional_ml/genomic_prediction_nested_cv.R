#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(G2P))
suppressPackageStartupMessages(library(data.table))

# --- 1. Parameter parsing: keep the Python CLI shape ---
option_list <- list(
  make_option("--feature_file", type = "character", default = NULL,
              help = "Feature/genotype file, same as Python --feature_file"),
  make_option("--y_file", type = "character", default = NULL,
              help = "Phenotype file, same as Python --y_file"),
  make_option("--id_dir", type = "character", default = NULL,
              help = "Directory containing outer_fold_*_train/test_IDs.txt"),
  make_option("--model_type", type = "character", default = "all",
              help = "G2P model(s): 'all', comma-separated, or space-separated [default: %default]"),
  make_option("--output_dir", type = "character", default = "./g2p_results",
              help = "Output directory [default: %default]"),
  make_option("--n_trials", type = "integer", default = 0,
              help = "Accepted for Python CLI compatibility; not used by G2P"),
  make_option("--device", type = "character", default = "cpu",
              help = "Accepted for Python CLI compatibility; not used by G2P"),
  make_option("--outer_k", type = "integer", default = 5,
              help = "Number of outer folds to read from --id_dir [default: %default]"),
  make_option("--inner_k", type = "integer", default = 3,
              help = "Number of inner folds for reference inner PCC [default: %default]"),
  make_option("--seed", type = "integer", default = 123,
              help = "Random seed for inner folds [default: %default]"),

  # Backward-compatible aliases from the old R interface.
  make_option("--X_file", type = "character", default = NULL,
              help = "Deprecated alias of --feature_file"),
  make_option("--Y_file", type = "character", default = NULL,
              help = "Deprecated alias of --y_file")
)

opt <- parse_args(OptionParser(option_list = option_list))

coalesce_arg <- function(primary, fallback) {
  if (!is.null(primary)) primary else fallback
}

feature_file <- coalesce_arg(opt$feature_file, opt$X_file)
y_file <- coalesce_arg(opt$y_file, opt$Y_file)
id_dir <- opt$id_dir

if (is.null(feature_file)) {
  stop("Missing required argument: --feature_file (or deprecated --X_file)", call. = FALSE)
}
if (is.null(y_file)) {
  stop("Missing required argument: --y_file (or deprecated --Y_file)", call. = FALSE)
}
if (is.null(id_dir)) {
  stop("Missing required argument: --id_dir", call. = FALSE)
}
if (!file.exists(feature_file)) {
  stop(sprintf("Feature file not found: %s", feature_file), call. = FALSE)
}
if (!file.exists(y_file)) {
  stop(sprintf("Phenotype file not found: %s", y_file), call. = FALSE)
}
if (!dir.exists(id_dir)) {
  stop(sprintf("ID directory not found: %s", id_dir), call. = FALSE)
}
if (opt$outer_k < 1) {
  stop("--outer_k must be >= 1", call. = FALSE)
}
if (opt$inner_k < 0) {
  stop("--inner_k must be >= 0", call. = FALSE)
}
if (!dir.exists(opt$output_dir)) {
  dir.create(opt$output_dir, recursive = TRUE)
}

# Regression-oriented default set. The parser below still accepts every
# method supported by G2P, including classification methods if requested.
default_g2p_methods <- c(
  "BayesA", "BayesB", "BayesC", "BL", "BRR", "RKHS", "RRBLUP",
  "LASSO", "SPLS", "SVR", "RFR", "RR", "bigRR", "BRNN"
)
supported_g2p_methods <- c(default_g2p_methods, "SVC", "RFC")

parse_methods <- function(model_type) {
  if (is.null(model_type) || tolower(trimws(model_type)) == "all") {
    return(default_g2p_methods)
  }

  requested <- trimws(unlist(strsplit(model_type, "[,[:space:]]+")))
  requested <- requested[nzchar(requested)]
  method_map <- setNames(supported_g2p_methods, tolower(supported_g2p_methods))
  matched <- unname(method_map[tolower(requested)])

  if (any(is.na(matched))) {
    stop(sprintf(
      "Unsupported --model_type value(s): %s. Supported values: all,%s",
      paste(requested[is.na(matched)], collapse = ","),
      paste(supported_g2p_methods, collapse = ",")
    ), call. = FALSE)
  }

  unique(matched)
}

sanitize_prefix <- function(x) {
  prefix <- tolower(gsub("[^A-Za-z0-9]+", "_", x))
  gsub("^_+|_+$", "", prefix)
}

normalize_ids <- function(x) {
  x <- as.character(x)
  x <- sub("^\ufeff", "", x)
  trimws(x)
}

read_ids <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Fold ID file not found: %s", path), call. = FALSE)
  }
  ids <- fread(path, header = FALSE, colClasses = "character")[[1]]
  ids <- normalize_ids(ids)
  ids[nzchar(ids)]
}

make_folds <- function(y, k) {
  n <- length(y)
  if (k < 2) {
    stop("k must be >= 2", call. = FALSE)
  }
  if (k > n) {
    stop(sprintf("k=%d is larger than sample size n=%d", k, n), call. = FALSE)
  }

  idx <- seq_len(n)
  ordered_idx <- idx[order(y, runif(n), na.last = TRUE)]
  folds <- vector("list", k)

  for (start in seq(1, n, by = k)) {
    block <- ordered_idx[start:min(start + k - 1, n)]
    fold_ids <- sample(seq_len(k), length(block))
    for (i in seq_along(block)) {
      folds[[fold_ids[i]]] <- c(folds[[fold_ids[i]]], block[i])
    }
  }

  folds
}

calc_metrics <- function(y_true, y_pred) {
  ok <- is.finite(y_true) & is.finite(y_pred)
  y_true_ok <- y_true[ok]
  y_pred_ok <- y_pred[ok]

  if (length(y_true_ok) == 0) {
    return(c(PCC = NA_real_, MSE = NA_real_, R2 = NA_real_))
  }

  pcc <- NA_real_
  if (length(y_true_ok) > 1 &&
      stats::sd(y_true_ok) > 0 &&
      stats::sd(y_pred_ok) > 0) {
    pcc <- suppressWarnings(stats::cor(y_true_ok, y_pred_ok))
  }

  mse <- mean((y_true_ok - y_pred_ok)^2)
  denom <- sum((y_true_ok - mean(y_true_ok))^2)
  r2 <- if (denom > 0) {
    1 - sum((y_true_ok - y_pred_ok)^2) / denom
  } else {
    NA_real_
  }

  c(PCC = pcc, MSE = mse, R2 = r2)
}

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE)
}

format_summary <- function(summary_df) {
  transform(summary_df, Mean = sprintf("%.4f", Mean), SD = sprintf("%.4f", SD))
}

# --- 2. Load and align data ---
cat(sprintf("Loading data...\n  feature_file: %s\n  y_file: %s\n  id_dir: %s\n",
            feature_file, y_file, id_dir))

X_dt <- fread(feature_file, check.names = FALSE)
if (ncol(X_dt) < 2) {
  stop("Feature file must contain one ID column plus at least one feature column", call. = FALSE)
}
feature_ids <- normalize_ids(X_dt[[1]])
if (length(feature_ids) != nrow(X_dt)) {
  stop(sprintf(
    "Feature ID length mismatch: n_ids=%d n_rows=%d file=%s",
    length(feature_ids), nrow(X_dt), feature_file
  ), call. = FALSE)
}
if (anyDuplicated(feature_ids)) {
  stop("Feature file contains duplicated IDs", call. = FALSE)
}

X_raw <- as.data.frame(X_dt[, -1, with = FALSE], stringsAsFactors = FALSE, check.names = FALSE)
rownames(X_raw) <- feature_ids
rm(X_dt)

Y_dt <- fread(y_file, check.names = FALSE)
if (ncol(Y_dt) < 2) {
  stop("Phenotype file must contain one ID column plus at least one trait column", call. = FALSE)
}
phenotype_ids <- normalize_ids(Y_dt[[1]])
if (length(phenotype_ids) != nrow(Y_dt)) {
  stop(sprintf(
    "Phenotype ID length mismatch: n_ids=%d n_rows=%d file=%s",
    length(phenotype_ids), nrow(Y_dt), y_file
  ), call. = FALSE)
}
if (anyDuplicated(phenotype_ids)) {
  stop("Phenotype file contains duplicated IDs", call. = FALSE)
}
Y_raw <- as.data.frame(Y_dt[, -1, with = FALSE], stringsAsFactors = FALSE, check.names = FALSE)
Y_raw <- data.frame(ID = phenotype_ids, Y_raw, check.names = FALSE)
rm(Y_dt)

trait <- colnames(Y_raw)[2]
Y_raw[[trait]] <- as.numeric(Y_raw[[trait]])

common_ids <- feature_ids[feature_ids %in% phenotype_ids]
if (length(common_ids) == 0) {
  stop(sprintf(
    paste0(
      "No overlapping sample IDs between feature and phenotype files.\n",
      "Feature IDs: n=%d example=%s\n",
      "Phenotype IDs: n=%d example=%s"
    ),
    length(feature_ids),
    paste(utils::head(feature_ids, 5), collapse = ","),
    length(phenotype_ids),
    paste(utils::head(phenotype_ids, 5), collapse = ",")
  ), call. = FALSE)
}

X <- as.matrix(X_raw[common_ids, , drop = FALSE])
storage.mode(X) <- "numeric"
Y <- Y_raw[match(common_ids, Y_raw$ID), c("ID", trait), drop = FALSE]
row.names(Y) <- NULL

id_to_idx <- setNames(seq_along(common_ids), common_ids)
methods_list <- parse_methods(opt$model_type)
method_prefixes <- setNames(vapply(methods_list, sanitize_prefix, character(1)), methods_list)

cat(sprintf("[Data] matched IDs=%d features=%d trait=%s\n", nrow(Y), ncol(X), trait))
cat(sprintf("[Models] %s\n", paste(methods_list, collapse = ", ")))
cat(sprintf(
  "[CLI] --n_trials=%s and --device=%s are accepted for Python compatibility and ignored by G2P.\n",
  opt$n_trials, opt$device
))

# --- 3. Nested CV: pass all requested methods to G2P together ---
set.seed(opt$seed)
all_metrics <- list()
all_inner_details <- list()
all_matrix_predictions <- list()
predictions_by_method <- setNames(vector("list", length(methods_list)), methods_list)
metrics_by_method <- setNames(vector("list", length(methods_list)), methods_list)

cat(sprintf("Starting G2P nested CV: %d outer folds x %d inner folds\n",
            opt$outer_k, opt$inner_k))

for (fold in seq_len(opt$outer_k)) {
  cat(sprintf("\n>>> Outer Fold %d/%d <<<\n", fold, opt$outer_k))

  train_path <- file.path(id_dir, sprintf("outer_fold_%d_train_IDs.txt", fold))
  test_path <- file.path(id_dir, sprintf("outer_fold_%d_test_IDs.txt", fold))
  train_ids <- read_ids(train_path)
  test_ids <- read_ids(test_path)

  missing_train <- setdiff(train_ids, names(id_to_idx))
  missing_test <- setdiff(test_ids, names(id_to_idx))
  if (length(missing_train) > 0 || length(missing_test) > 0) {
    stop(sprintf(
      "Fold %d has IDs missing from aligned feature/phenotype data. train missing=%d, test missing=%d",
      fold, length(missing_train), length(missing_test)
    ), call. = FALSE)
  }

  outer_train_idx <- unname(id_to_idx[train_ids])
  outer_test_idx <- unname(id_to_idx[test_ids])

  sds <- apply(X[outer_train_idx, , drop = FALSE], 2, stats::sd, na.rm = TRUE)
  keep_cols <- is.finite(sds) & sds > 0
  if (!any(keep_cols)) {
    stop(sprintf("No non-constant feature columns remain in fold %d", fold), call. = FALSE)
  }
  X_filtered <- X[, keep_cols, drop = FALSE]

  inner_pcc_by_method <- setNames(rep(NA_real_, length(methods_list)), methods_list)
  if (opt$inner_k >= 2) {
    cat(sprintf("  Inner CV with %d folds...\n", opt$inner_k))
    inner_folds <- make_folds(Y[outer_train_idx, trait], k = opt$inner_k)
    inner_pcc_lists <- setNames(vector("list", length(methods_list)), methods_list)

    for (inner_fold in seq_along(inner_folds)) {
      inner_val_sub <- inner_folds[[inner_fold]]
      curr_inner_val <- outer_train_idx[inner_val_sub]
      curr_inner_train <- setdiff(outer_train_idx, curr_inner_val)

      res_inner <- G2P(
        markers = X_filtered,
        data = Y,
        trait = trait,
        modelMethods = methods_list,
        trainIdx = curr_inner_train,
        predIdx = curr_inner_val
      )

      for (method in methods_list) {
        inner_metrics <- calc_metrics(
          y_true = as.numeric(res_inner[, "realPhenScore"]),
          y_pred = as.numeric(res_inner[, method])
        )
        inner_pcc_lists[[method]] <- c(inner_pcc_lists[[method]], inner_metrics[["PCC"]])
      }
    }

    inner_pcc_by_method <- sapply(inner_pcc_lists, function(x) {
      value <- mean(x, na.rm = TRUE)
      if (is.nan(value)) NA_real_ else value
    })
  }

  cat("  Outer G2P fit/predict for all methods...\n")
  res_outer <- G2P(
    markers = X_filtered,
    data = Y,
    trait = trait,
    modelMethods = methods_list,
    trainIdx = outer_train_idx,
    predIdx = outer_test_idx
  )

  res_outer <- as.matrix(res_outer)
  if (is.null(rownames(res_outer))) {
    rownames(res_outer) <- test_ids
  }

  outer_matrix_df <- data.frame(
    ID = rownames(res_outer),
    Fold = fold,
    as.data.frame(res_outer, check.names = FALSE),
    check.names = FALSE
  )
  write_tsv(
    outer_matrix_df,
    file.path(opt$output_dir, sprintf("g2p_outer_fold_%d_predictions.txt", fold))
  )
  all_matrix_predictions[[length(all_matrix_predictions) + 1]] <- outer_matrix_df

  for (method in methods_list) {
    metrics <- calc_metrics(
      y_true = as.numeric(res_outer[, "realPhenScore"]),
      y_pred = as.numeric(res_outer[, method])
    )

    method_metrics <- data.frame(
      Fold = fold,
      PCC = unname(metrics[["PCC"]]),
      MSE = unname(metrics[["MSE"]]),
      R2 = unname(metrics[["R2"]]),
      check.names = FALSE
    )
    method_predictions <- data.frame(
      ID = rownames(res_outer),
      Fold = fold,
      y_true = as.numeric(res_outer[, "realPhenScore"]),
      y_pred = as.numeric(res_outer[, method]),
      check.names = FALSE
    )
    inner_detail <- data.frame(
      Method = method,
      Fold = fold,
      Inner_Avg_PCC = unname(inner_pcc_by_method[[method]]),
      check.names = FALSE
    )

    metrics_by_method[[method]][[length(metrics_by_method[[method]]) + 1]] <- method_metrics
    predictions_by_method[[method]][[length(predictions_by_method[[method]]) + 1]] <- method_predictions
    all_metrics[[length(all_metrics) + 1]] <- cbind(Method = method, method_metrics)
    all_inner_details[[length(all_inner_details) + 1]] <- inner_detail

    cat(sprintf("    %s: R2=%.4f, PCC=%.4f\n",
                method, metrics[["R2"]], metrics[["PCC"]]))
  }
}

# --- 4. Per-model Python-style exports ---
for (method in methods_list) {
  prefix <- method_prefixes[[method]]
  method_metrics_df <- do.call(rbind, metrics_by_method[[method]])
  method_predictions_df <- do.call(rbind, predictions_by_method[[method]])

  write.csv(method_metrics_df,
            file.path(opt$output_dir, sprintf("%s_metrics.csv", prefix)),
            row.names = FALSE)
  write.csv(method_predictions_df,
            file.path(opt$output_dir, sprintf("%s_predictions.csv", prefix)),
            row.names = FALSE)

  for (fold in seq_len(opt$outer_k)) {
    fold_predictions <- method_predictions_df[method_predictions_df$Fold == fold, , drop = FALSE]
    write.csv(
      fold_predictions,
      file.path(opt$output_dir, sprintf("%s_outer_fold_%d_predictions.csv", prefix, fold)),
      row.names = FALSE
    )
  }

  summary_df <- data.frame(
    Method = toupper(method),
    Metric = c("PCC", "MSE", "R2"),
    Mean = c(mean(method_metrics_df$PCC, na.rm = TRUE),
             mean(method_metrics_df$MSE, na.rm = TRUE),
             mean(method_metrics_df$R2, na.rm = TRUE)),
    SD = c(stats::sd(method_metrics_df$PCC, na.rm = TRUE),
           stats::sd(method_metrics_df$MSE, na.rm = TRUE),
           stats::sd(method_metrics_df$R2, na.rm = TRUE)),
    row.names = NULL,
    check.names = FALSE
  )
  write_tsv(format_summary(summary_df),
            file.path(opt$output_dir, sprintf("%s_nested_summary.txt", prefix)))
}

# --- 5. Combined exports and terminal report ---
full_metrics <- do.call(rbind, all_metrics)
inner_details <- do.call(rbind, all_inner_details)
matrix_predictions <- do.call(rbind, all_matrix_predictions)

combined_summary <- do.call(rbind, lapply(methods_list, function(method) {
  df <- full_metrics[full_metrics$Method == method, , drop = FALSE]
  data.frame(
    Method = method,
    Metric = c("PCC", "MSE", "R2"),
    Mean = c(mean(df$PCC, na.rm = TRUE),
             mean(df$MSE, na.rm = TRUE),
             mean(df$R2, na.rm = TRUE)),
    SD = c(stats::sd(df$PCC, na.rm = TRUE),
           stats::sd(df$MSE, na.rm = TRUE),
           stats::sd(df$R2, na.rm = TRUE)),
    row.names = NULL,
    check.names = FALSE
  )
}))
row.names(combined_summary) <- NULL

legacy_summary <- do.call(rbind, lapply(methods_list, function(method) {
  df <- combined_summary[combined_summary$Method == method, , drop = FALSE]
  data.frame(
    Method = method,
    PCC_Mean = df$Mean[df$Metric == "PCC"],
    PCC_SD = df$SD[df$Metric == "PCC"],
    MSE_Mean = df$Mean[df$Metric == "MSE"],
    MSE_SD = df$SD[df$Metric == "MSE"],
    R2_Mean = df$Mean[df$Metric == "R2"],
    R2_SD = df$SD[df$Metric == "R2"],
    row.names = NULL,
    check.names = FALSE
  )
}))

write.csv(full_metrics,
          file.path(opt$output_dir, "g2p_all_metrics.csv"),
          row.names = FALSE)
write.csv(merge(full_metrics, inner_details, by = c("Method", "Fold"), all.x = TRUE),
          file.path(opt$output_dir, "nested_cv_details_per_fold.csv"),
          row.names = FALSE)
write.csv(inner_details,
          file.path(opt$output_dir, "g2p_inner_details.csv"),
          row.names = FALSE)
write_tsv(matrix_predictions,
          file.path(opt$output_dir, "g2p_predictions.txt"))
write_tsv(format_summary(combined_summary),
          file.path(opt$output_dir, "g2p_all_nested_summary.txt"))
write_tsv(legacy_summary,
          file.path(opt$output_dir, "nested_cv_summary_metrics.txt"))

cat("\n==================================================\n")
cat("FINAL G2P NESTED CROSS-VALIDATION SUMMARY\n")
cat("==================================================\n")
print(transform(combined_summary, Mean = round(Mean, 4), SD = round(SD, 4)),
      row.names = FALSE)
cat("==================================================\n")

save(full_metrics, inner_details, matrix_predictions, combined_summary,
     file = file.path(opt$output_dir, "G2P_Nested_Data.Rdata"))
cat(sprintf("All results saved to: %s\n", opt$output_dir))
