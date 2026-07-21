#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
})

options_list <- list(
  make_option("--result_dir", type = "character", help = "Directory containing chr*/ outputs"),
  make_option("--chromosomes", type = "character", default = "1-12"),
  make_option("--output_expression", type = "character", help = "Merged predicted-expression TSV"),
  make_option("--output_metrics", type = "character", help = "Merged per-gene metric TSV")
)
opt <- parse_args(OptionParser(option_list = options_list))
required <- c("result_dir", "output_expression", "output_metrics")
missing <- required[vapply(required, function(name) is.null(opt[[name]]), logical(1))]
if (length(missing)) stop("Missing options: ", paste(paste0("--", missing), collapse = ", "), call. = FALSE)

expand_chromosomes <- function(text) {
  tokens <- trimws(unlist(strsplit(text, ",")))
  unique(unlist(lapply(tokens[nzchar(tokens)], function(token) {
    if (grepl("^[0-9]+-[0-9]+$", token)) {
      bounds <- as.integer(unlist(strsplit(token, "-")))
      as.character(seq(bounds[1], bounds[2]))
    } else token
  })))
}

chromosomes <- expand_chromosomes(opt$chromosomes)
reference_ids <- NULL
expression_blocks <- list()
metric_blocks <- list()

for (chromosome in chromosomes) {
  chromosome_dir <- file.path(opt$result_dir, paste0("chr", chromosome))
  expression_path <- file.path(chromosome_dir, "Predict_Expre.tsv")
  metric_path <- file.path(chromosome_dir, "Predict_metrics.tsv")
  if (!file.exists(expression_path) || !file.exists(metric_path)) {
    stop("Missing outputs for chromosome ", chromosome, call. = FALSE)
  }
  expression <- fread(expression_path, check.names = FALSE)
  if (!("ID" %in% names(expression))) stop("Missing ID column in ", expression_path, call. = FALSE)
  ids <- as.character(expression$ID)
  if (anyDuplicated(ids)) stop("Duplicated IDs in ", expression_path, call. = FALSE)
  if (is.null(reference_ids)) {
    reference_ids <- ids
  } else {
    if (!setequal(reference_ids, ids)) stop("Sample set differs in ", expression_path, call. = FALSE)
    expression <- expression[match(reference_ids, ids)]
  }
  genes <- setdiff(names(expression), "ID")
  duplicated_genes <- intersect(genes, unlist(lapply(expression_blocks, names)))
  if (length(duplicated_genes)) stop("Genes duplicated across chromosomes: ",
                                    paste(head(duplicated_genes, 10), collapse = ","), call. = FALSE)
  expression_blocks[[chromosome]] <- expression[, ..genes]
  metrics <- fread(metric_path, check.names = FALSE)
  metrics[, Chromosome := chromosome]
  setcolorder(metrics, c("Chromosome", setdiff(names(metrics), "Chromosome")))
  metric_blocks[[chromosome]] <- metrics
}

merged_expression <- data.table(ID = reference_ids)
if (length(expression_blocks)) {
  merged_expression <- cbind(merged_expression, do.call(cbind, unname(expression_blocks)))
}
dir.create(dirname(opt$output_expression), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(opt$output_metrics), recursive = TRUE, showWarnings = FALSE)
fwrite(merged_expression, opt$output_expression, sep = "\t", quote = FALSE)
fwrite(rbindlist(metric_blocks, use.names = TRUE, fill = TRUE), opt$output_metrics, sep = "\t", quote = FALSE)
message("Merged samples=", nrow(merged_expression), "; genes=", ncol(merged_expression) - 1)
