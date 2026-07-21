#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
})

options_list <- list(
  make_option("--input_dir", type = "character", help = "One tissue's chromosome-output directory"),
  make_option("--chromosomes", type = "character", default = "1-22",
              help = "Chromosome list/ranges, e.g. 1-22 or 1,2,X"),
  make_option("--file_template", type = "character", default = "en_chr{chr}_prediction.txt",
              help = "Filename pattern relative to input_dir"),
  make_option("--id_columns", type = "character", default = "FID,IID",
              help = "Comma-separated identifier columns"),
  make_option("--output", type = "character", help = "Merged sample-by-gene TSV"),
  make_option("--allow_missing_chromosomes", action = "store_true", default = FALSE)
)
opt <- parse_args(OptionParser(option_list = options_list))

if (is.null(opt$input_dir) || is.null(opt$output)) {
  stop("--input_dir and --output are required", call. = FALSE)
}

expand_chromosomes <- function(text) {
  tokens <- trimws(unlist(strsplit(text, ",")))
  out <- character()
  for (token in tokens[nzchar(tokens)]) {
    if (grepl("^[0-9]+-[0-9]+$", token)) {
      bounds <- as.integer(unlist(strsplit(token, "-")))
      out <- c(out, as.character(seq(bounds[1], bounds[2])))
    } else {
      out <- c(out, token)
    }
  }
  unique(out)
}

requested_ids <- trimws(unlist(strsplit(opt$id_columns, ",")))
chromosomes <- expand_chromosomes(opt$chromosomes)
id_reference <- NULL
feature_tables <- list()
manifest <- list()

for (chromosome in chromosomes) {
  filename <- gsub("\\{chr\\}", chromosome, opt$file_template)
  path <- file.path(opt$input_dir, filename)
  if (!file.exists(path)) {
    if (opt$allow_missing_chromosomes) {
      warning("Skipping missing chromosome file: ", path)
      next
    }
    stop("Missing chromosome file: ", path, call. = FALSE)
  }

  current <- fread(path, check.names = FALSE)
  id_columns <- requested_ids[requested_ids %in% names(current)]
  if (length(id_columns) == 0) {
    stop(
      "None of --id_columns were found in ", path,
      "; columns are: ", paste(names(current), collapse = ","),
      call. = FALSE
    )
  }
  current_ids <- current[, ..id_columns]
  key <- do.call(paste, c(lapply(current_ids, as.character), sep = "\r"))
  if (anyDuplicated(key)) {
    stop("Duplicated sample identifiers in ", path, call. = FALSE)
  }

  if (is.null(id_reference)) {
    id_reference <- copy(current_ids)
    reference_key <- key
  } else {
    if (!setequal(reference_key, key)) {
      stop("Sample set differs in ", path, call. = FALSE)
    }
    current <- current[match(reference_key, key)]
  }

  feature_columns <- setdiff(names(current), id_columns)
  if (length(feature_columns) == 0) {
    warning("No predicted genes in ", path)
    next
  }
  duplicated_features <- intersect(feature_columns, unlist(lapply(feature_tables, names)))
  if (length(duplicated_features) > 0) {
    stop("Predicted genes occur in multiple chromosome files: ",
         paste(head(duplicated_features, 10), collapse = ","), call. = FALSE)
  }
  feature_tables[[chromosome]] <- current[, ..feature_columns]
  manifest[[chromosome]] <- data.table(
    Chromosome = chromosome,
    Source = normalizePath(path),
    Samples = nrow(current),
    PredictedGenes = length(feature_columns)
  )
}

if (is.null(id_reference) || length(feature_tables) == 0) {
  stop("No chromosome predictions were available to merge", call. = FALSE)
}

merged <- cbind(id_reference, do.call(cbind, unname(feature_tables)))
dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
fwrite(merged, opt$output, sep = "\t", quote = FALSE)
fwrite(rbindlist(manifest), paste0(opt$output, ".manifest.tsv"), sep = "\t", quote = FALSE)
message("Merged samples=", nrow(merged), "; genes=", ncol(merged) - ncol(id_reference),
        "; output=", opt$output)
