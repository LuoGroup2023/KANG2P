# Dependency groups

- `all.txt`: Python dependencies for disease models, DualKAN, plotting, and
  tree baselines.
- Human predicted expression additionally requires an external MetaXcan or
  PrediXcan checkout and compatible tissue model databases.
- Crop predicted expression requires R packages `optparse`, `data.table`,
  `progress`, `plink2R`, `glue`, and `glmnet`, plus PLINK 1.9.
- Traditional plant genomic prediction requires R package `G2P`.

External model databases, cohort data, genome annotations, liftover chains,
and PLINK binaries are intentionally not vendored.
