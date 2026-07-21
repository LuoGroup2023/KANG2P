# 06. DualKAN gated fusion

The model has independent genotype and predicted-expression encoders,
learnable modality gates, interaction features, reconstruction losses, and a
spline or Fourier KAN prediction head. Feature selection and scaling are fit
inside training folds.

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
