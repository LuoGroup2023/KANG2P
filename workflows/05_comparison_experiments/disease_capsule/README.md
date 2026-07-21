# DiseaseCapsule baseline

This is one shared DiseaseCapsule implementation for ALS/AD and G/PE/PP
inputs. Repeat `--features` for early concatenation of aligned modalities.

```bash
bash run_comparison.sh \
  --features pkl:/secure/ALS/genotype.pkl \
  --features tsv:/secure/ALS/predicted_expression.tsv \
  --labels pkl:/secure/ALS/labels.pkl \
  --split-dir /secure/ALS/cv_splits_5fold \
  --output-dir ../../../outputs/disease_capsule/als_g_pe \
  --prefix als_g_pe
```

Optuna tuning uses inner stratified CV. Imputation/scaling is fitted within
each inner training split. Set `CAPSNET_TRIALS=0` for a fast fixed-config
check, and add `--save-models` only when checkpoints are needed.
