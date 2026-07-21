# 03. Predicted proteome

The audited workspace contains a 1,473-protein AD prediction matrix and
downstream PP experiments, but not the upstream pQTL/protein-weight scoring
program or its model weights. Those external assets are therefore a known
reproducibility gap and are not reconstructed by guesswork here.

`prepare_predicted_proteome.py` covers the reproducible boundary available in
the workspace: sample alignment, model/protein allow-listing, missingness and
variance QC, training-reference imputation, optional standardization, and
transform metadata export.

```bash
python prepare_predicted_proteome.py \
  --prediction-table /secure/AD/raw_protein_predictions.tsv \
  --fit-sample-ids /secure/AD/outer_training_ids.txt \
  --standardize \
  --output ../../outputs/predicted_proteome/ad_pp.tsv
```

After preparation, set `PP_TABLE`, `LABEL_PKL`, and `SPLIT_DIR`, then run
`run_pp_nested_cv.sh`. It calls the shared comparison workflow so
DiseaseCapsule and traditional ML receive identical inputs and outer folds.

Before public release, add the exact upstream tool version, weight source,
genome build, allele-harmonization procedure, and failed-model allow-list used
to create the historical matrix.
