#!/usr/bin/env Rscript

# Select predicted-expression features using only one outer-training fold.
suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
})

options_list <- list(
  make_option("--predicted_expression", type = "character", help = "Merged PrediXcan TSV"),
  make_option("--labels", type = "character", help = "Sample label TSV/CSV"),
  make_option("--train_ids", type = "character", help = "One-column outer-training sample IDs"),
  make_option("--sample_id_column", type = "character", default = "IID"),
  make_option("--label_id_column", type = "character", default = "FID"),
  make_option("--label_column", type = "character", default = "Pheno"),
  make_option("--control_value", type = "character", default = "1"),
  make_option("--case_value", type = "character", default = "2"),
  make_option("--p_threshold", type = "double", default = 0.05),
  make_option("--fdr_threshold", type = "double", default = NA_real_),
  make_option("--output_dir", type = "character", help = "Fold-specific output directory")
)
opt <- parse_args(OptionParser(option_list = options_list))

required <- c("predicted_expression", "labels", "train_ids", "output_dir")
missing <- required[vapply(required, function(name) is.null(opt[[name]]), logical(1))]
if (length(missing) > 0) {
  stop("Missing required options: ", paste(paste0("--", missing), collapse = ", "), call. = FALSE)
}
if (!(opt$p_threshold > 0 && opt$p_threshold <= 1)) {
  stop("--p_threshold must be in (0, 1]", call. = FALSE)
}

read_auto <- function(path) {
  fread(path, sep = "auto", check.names = FALSE)
}
expression <- read_auto(opt$predicted_expression)
labels <- read_auto(opt$labels)
train_ids <- trimws(as.character(fread(opt$train_ids, header = FALSE)[[1]]))
train_ids <- unique(train_ids[nzchar(train_ids)])

for (column in c(opt$sample_id_column)) {
  if (!(column %in% names(expression))) stop("Missing expression ID column: ", column, call. = FALSE)
}
for (column in c(opt$label_id_column, opt$label_column)) {
  if (!(column %in% names(labels))) stop("Missing label column: ", column, call. = FALSE)
}

expression_ids <- as.character(expression[[opt$sample_id_column]])
label_ids <- as.character(labels[[opt$label_id_column]])
if (anyDuplicated(expression_ids) || anyDuplicated(label_ids)) {
  stop("Expression or label table contains duplicated sample IDs", call. = FALSE)
}
missing_train <- setdiff(train_ids, intersect(expression_ids, label_ids))
if (length(missing_train) > 0) {
  stop(length(missing_train), " training IDs are absent from expression/label tables", call. = FALSE)
}

train_expression <- expression[match(train_ids, expression_ids)]
train_labels <- as.character(labels[[opt$label_column]][match(train_ids, label_ids)])
if (!all(c(opt$control_value, opt$case_value) %in% train_labels)) {
  stop("Both control and case values must occur in the outer-training fold", call. = FALSE)
}

id_candidates <- intersect(c("FID", "IID"), names(expression))
gene_columns <- setdiff(names(expression), id_candidates)
if (length(gene_columns) == 0) stop("No predicted-expression columns found", call. = FALSE)

test_one <- function(gene) {
  values <- suppressWarnings(as.numeric(train_expression[[gene]]))
  control <- values[train_labels == opt$control_value]
  case <- values[train_labels == opt$case_value]
  control <- control[is.finite(control)]
  case <- case[is.finite(case)]
  p_value <- if (length(control) >= 2 && length(case) >= 2) {
    tryCatch(wilcox.test(control, case, exact = FALSE)$p.value, error = function(e) NA_real_)
  } else {
    NA_real_
  }
  data.table(
    Gene = gene,
    PValue = p_value,
    ControlMean = if (length(control)) mean(control) else NA_real_,
    CaseMean = if (length(case)) mean(case) else NA_real_,
    NControl = length(control),
    NCase = length(case)
  )
}

tests <- rbindlist(lapply(gene_columns, test_one))
tests[, FDR := p.adjust(PValue, method = "fdr")]
keep <- is.finite(tests$PValue) & tests$PValue < opt$p_threshold
if (is.finite(opt$fdr_threshold)) {
  keep <- keep & is.finite(tests$FDR) & tests$FDR < opt$fdr_threshold
}
selected <- tests$Gene[keep]

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(tests[order(PValue)], file.path(opt$output_dir, "wilcox_training_only.tsv"), sep = "\t")
fwrite(expression[, c(id_candidates, selected), with = FALSE],
       file.path(opt$output_dir, "predicted_expression_selected.tsv"), sep = "\t")
fwrite(data.table(Metric = c("TrainingSamples", "AllGenes", "SelectedGenes"),
                  Value = c(length(train_ids), length(gene_columns), length(selected))),
       file.path(opt$output_dir, "selection_summary.tsv"), sep = "\t")
message("Training samples=", length(train_ids), "; selected genes=", length(selected),
        "/", length(gene_columns))
