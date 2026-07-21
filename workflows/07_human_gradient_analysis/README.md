# 07. Human gradient analysis

The workflow operates on human ALS genotype Gene-PCA features and predicted
expression. It provides:

- `als_g_gene_interpretability.py`: univariate, optional SGD, G-only KAN, and
  DualBranch genotype-gradient aggregation;
- `als_g_add_ig_lime_rankings.py`: integrated gradients and grouped
  LIME-style rankings;
- `als_gpe_dualbranch_interpretability.py`: joint G/PE gradient, gradient ×
  input, integrated gradients, encoder weights, and grouped local surrogates.

```bash
python als_gpe_dualbranch_interpretability.py \
  --g-pkl /secure/ALS/genotype_gene_pca.pkl \
  --pe-txt /secure/ALS/predicted_expression.tsv \
  --dualbranch-checkpoint /secure/checkpoints/als_g_pe.pt \
  --out-dir ../../outputs/human_gradient/ALS/g_pe
```

The G-only script also emits an explicitly labelled prior-aware SLC1A2 score.
Use the raw model-only columns for unbiased genome-wide interpretation.
