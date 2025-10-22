# Chuqiao & Zichun Readmission Pipeline

This folder contains a light wrapper around Yuchen Zhou's training pipeline that
forces the experiment to use only the top 20 features from the pre-computed
LASSO importance file. The wrapper keeps all generated artifacts and reports in
this directory so it does not interfere with the original pipeline outputs.

## Layout

- `config_top20.yaml` – configuration file that mirrors Yuchen's defaults but
  tightens feature selection to the top 20 features and limits the run to
  logistic regression for quick iteration.
- `run_top20.py` – helper script that launches the original training logic with
  the custom configuration and writes outputs to `artifacts/` and `reports/`
  under this directory.
- `artifacts/`, `reports/` – created automatically after the training run.

## Running the experiment

From the project root (or this folder), execute:

```bash
python run_top20.py
```

The script delegates to `YuchenZhou_Pipeline/training/src/train.py`, so all
preprocessing, model training, and evaluation follow Yuchen's implementation.
Results (metrics CSV, plots, and the trained logistic regression model) will
appear in `reports/` and `artifacts/` inside this folder.
