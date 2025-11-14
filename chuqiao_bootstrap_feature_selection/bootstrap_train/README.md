# Bootstrap-Aware Training Helper

This folder links the bootstrap feature stability study with Yuchen's
end-to-end training pipeline. The `run_bootstrap_training.py` script:

1. Reads `bootstrap_results/feature_importance_ranked.csv`.
2. Keeps the top-K raw features by bootstrap stability (default = 50).
3. Converts them into the `feature/coef/importance` format that Yuchen's
   pipeline expects.
4. Writes a derived config pointing to those features and to local artifact
   directories.
5. Launches `YuchenZhou_jiaqi_Pipeline/training/src/train.py` with that config.

## Usage

From the project root:

```bash
python chuqiao_bootstrap_feature_selection/bootstrap_train/run_bootstrap_training.py \
    --top-k 50 \
    --models logistic,rf,xgb
```

Flags:
- `--top-k`: number of bootstrap-ranked features to keep (default 50).
- `--models`: comma-separated list passed to the training config when
  `--model all` is used (defaults to `logistic,rf,xgb,transformer`).
- `--prepare-only`: stop after generating the feature list and config.
- `--verbose`: print the selected features and their bootstrap stats.

Artifacts and reports are written under this folder:
- `artifacts/bootstrap_top_features.csv`: feature importance file consumed by
  Yuchen's pipeline.
- `generated_config.yaml`: derived config used for training.
- `artifacts/` and `reports/`: model outputs and evaluation reports produced by
  the pipeline.

The script automatically reuses Yuchen's existing `config.yaml` for everything
else (data path, preprocessing, hyperparameters, etc.), so any updates made
there propagate here as well.

## Transformer-Only Debugging

If you want to step through just the transformer portion (without the other
models or try/except wrappers), use:

```bash
python chuqiao_bootstrap_feature_selection/bootstrap_train/run_transformer_debug.py \
    --top-k 30 \
    --ranking-path ./chuqiao_bootstrap_feature_selection/bootstrap_results/feature_importance_ranked.csv \
    --base-config ./YuchenZhou_jiaqi_Pipeline/training/config.yaml \
    --verbose
```

This helper regenerates the top-K feature file (unless `--skip-prep` is set),
builds a config that only lists the transformer model, and then reproduces the
transformer branch of Yuchen's pipeline in a standalone script. Because it calls
the underlying `train_pytorch_model` directly, any exception will surface in the
terminal/VS Code debugger, making it much easier to inspect tensors, batches,
and forward passes without waiting for the rest of the pipeline.
