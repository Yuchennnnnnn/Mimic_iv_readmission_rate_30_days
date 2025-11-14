# Bootstrap Feature Stability 

This helper isolates the "bootstrap L1 selection" idea from the feedback so we
can measure how stable each feature is across resampled datasets without
changing the existing pipelines.

## What it does

1. Loads any cleaned readmission dataset (default: `../cleaned_data.csv`).
2. Optionally restricts the columns to a feature list (e.g., our LASSO file).
3. Fits an L1-regularized logistic regression on many bootstrap resamples.
4. Tracks how often every encoded indicator receives a non-zero coefficient and
   aggregates those counts back to the raw feature names.
5. Writes CSV reports plus a short text summary under the chosen output folder.

## Usage

```bash
cd chuqiao_bootstrap_feature_selection
python bootstrap_feature_selection.py \
    --data-path ../cleaned_data.csv \
    --feature-file ../YuchenZhou_jiaqi_Pipeline/Feature_Importance_by_Coef.csv \
    --target-column readmit_label \
    --n-bootstrap 200 \
    --sample-fraction 0.8 \
    --c-value 0.1 \
    --output-dir bootstrap_results
```

Key flags:
- `--feature-file`: CSV/TXT with a `feature` column (our LASSO export works).
  Omit this flag to use every column except the target.
- `--sample-fraction`: Size of each bootstrap sample relative to the full
  dataset (with replacement). Defaults to 0.7.
- `--n-bootstrap`: Number of resamples. More iterations give smoother
  frequencies at the cost of runtime.
- `--coef-threshold`: Absolute coefficient cutoff for counting a feature as
  "selected" (default = 1e-6 so anything non-zero counts).
- `--c-value`: Inverse regularization strength for the L1 logistic model.

## Outputs

The script creates (or reuses) the folder given to `--output-dir` and writes:

- `encoded_feature_stability.csv` – every encoded indicator (e.g.,
  `cat__admission_type_EMERGENCY`) with selection counts, percentages, and the
  bootstrap mean/std of its coefficient.
- `raw_feature_stability.csv` – grouped view that collapses the encoded
  indicators back to their original columns and reports:
  - number of encoded components
  - mean and max selection percentage across those components
  - mean absolute coefficient magnitude
- `summary.txt` – run settings plus the top 10 raw features by mean selection
  percentage for a quick glance.

## Interpreting the results

- **High selection % (close to 1.0)**: the feature survives almost every
  bootstrap sample → very stable.
- **Low but non-zero selection %**: the feature only matters for certain
  subsets → consider whether to keep it or merge categories.
- **Large coef std**: indicates instability in magnitude/direction, signaling
  potential multicollinearity or noisy signal.

These tables plug directly into the professor's suggestions:
- Shows whether our top ~100 features remain stable when the cohort changes.
- Highlights candidates for correlation-based pruning (low stability, high
  collinearity).
- Provides evidence when we aggregate or drop sparse categories to control the
  feature space explosion from one-hot encoding.
