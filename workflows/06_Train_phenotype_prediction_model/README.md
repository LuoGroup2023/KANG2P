# 06. Train phenotype prediction model

The DualKAN gated-fusion phenotype-prediction model has independent genotype
and predicted-expression encoders,
learnable modality gates, interaction features, reconstruction losses, and a
spline or Fourier KAN prediction head. Feature selection and scaling are fit
inside training folds.

## Bundled reviewer demo

From the repository root, run the core model on the included synthetic data:

```bash
pip install -r requirements/dualkan.txt
```

Dependency installation typically takes 5–15 minutes with prebuilt wheels and
a stable broadband connection, or 15–30 minutes on a slower connection.
PyTorch is the largest download; cached environments are usually ready in
under a minute.

```bash
python workflows/06_Train_phenotype_prediction_model/dualkan_gated_fusion.py \
  --data_root demo_data/plant \
  --datasets Rice18K \
  --trait Rice18K:Grain_yield \
  --output_dir outputs/reviewer_demo \
  --device cpu \
  --disable_amp \
  --force_random_search \
  --head_type fourier \
  --smoke
```

This runs one short outer-fold check without controlled data, external model
databases, or a GPU. See [`../../demo_data/README.md`](../../demo_data/README.md)
for the fixture contents and limitations.

```bash
DATA_ROOT=/secure/plant \
GPU_LIST=0,1,2,3 \
HEAD_TYPE=fourier \
N_TRIALS=12 \
bash run_all_crops.sh
```

Set `TASK_LIST=Rice18K:Grain_yield` for one task or pass additional Python
arguments after the launcher command. Results include fold metrics,
predictions, selected features, checkpoints, and an aggregated run summary.
