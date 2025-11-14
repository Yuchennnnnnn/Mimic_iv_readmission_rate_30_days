#!/usr/bin/env python3
"""Bootstrap-based feature selection stability analysis.

This script repeatedly fits an L1-regularized logistic regression model on
bootstrap resamples of the cleaned readmission dataset. After every fit it
records which encoded features receive non-zero coefficients and aggregates the
results so you can see how frequently each feature (and each encoded indicator)
appears across the resamples.

Example:
    python bootstrap_feature_selection.py \
        --data-path ../cleaned_data.csv \
        --feature-file ../YuchenZhou_jiaqi_Pipeline/Feature_Importance_by_Coef.csv \
        --output-dir bootstrap_outputs \
        --n-bootstrap 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _read_feature_list(path: Path) -> List[str]:
    """Load feature names from either a CSV (with a `feature` column) or text file."""

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" in df.columns:
            values = df["feature"].astype(str).tolist()
        else:
            values = df.iloc[:, 0].astype(str).tolist()
    else:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            values = [row[0] for row in reader if row]
    return [v.strip() for v in values if v and v.strip()]


def _build_preprocessor(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_overrides: Iterable[str] | None = None,
) -> tuple[ColumnTransformer, List[str], List[str]]:
    """Create a ColumnTransformer for numeric + categorical columns."""

    if categorical_overrides is not None:
        categorical_cols = [col for col in feature_cols if col in categorical_overrides]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    else:
        categorical_cols = [
            col
            for col in feature_cols
            if df[col].dtype == "object"
            or df[col].dtype.name == "category"
        ]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No features left after preprocessing step.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    preprocessor.fit(df[feature_cols])
    return preprocessor, numeric_cols, categorical_cols


def _raw_feature_name(encoded_name: str, categorical_cols: Sequence[str]) -> str:
    """Map an encoded ColumnTransformer feature name back to the raw column."""

    if "__" in encoded_name:
        encoded_name = encoded_name.split("__", 1)[1]

    for cat in categorical_cols:
        prefix = f"{cat}_"
        if encoded_name.startswith(prefix):
            return cat
    return encoded_name


def run_bootstrap(
    data_path: Path,
    target_column: str,
    feature_file: Path | None,
    output_dir: Path,
    n_bootstrap: int,
    sample_fraction: float,
    c_value: float,
    coef_threshold: float,
    max_iter: int,
    random_state: int | None,
) -> None:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in the dataset.")

    candidate_features = [col for col in df.columns if col != target_column]
    if feature_file is not None:
        requested = _read_feature_list(feature_file)
        # Keep only those present in the dataset
        feature_cols = [c for c in requested if c in candidate_features]
        missing = sorted(set(requested) - set(feature_cols))
        if missing:
            print(
                f"[WARN] Ignoring {len(missing)} requested feature(s) that were not found in the dataset.",
                file=sys.stderr,
            )
    else:
        feature_cols = candidate_features

    if not feature_cols:
        raise ValueError("No usable feature columns were found.")

    sub_df = df[feature_cols + [target_column]].dropna()
    if sub_df.empty:
        raise ValueError("All rows were dropped after removing NaNs; nothing to train on.")

    y = sub_df[target_column].astype(int).to_numpy()
    X_df = sub_df[feature_cols]

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(X_df, feature_cols)
    feature_names = preprocessor.get_feature_names_out()
    encoded_to_raw = {
        name: _raw_feature_name(name, categorical_cols) for name in feature_names
    }

    X_full = preprocessor.transform(X_df)
    n_samples = X_full.shape[0]
    n_features = X_full.shape[1]

    rng = np.random.default_rng(random_state)
    coef_history = np.zeros((n_bootstrap, n_features), dtype=np.float64)
    selection_counts = np.zeros(n_features, dtype=np.int32)

    min_samples = max(1, int(sample_fraction * n_samples))

    print(
        f"Running {n_bootstrap} bootstrap iterations on {n_samples} rows "
        f"with sample fraction {sample_fraction:.2f} (~{min_samples} rows each)."
    )

    for i in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=min_samples)
        X_boot = X_full[indices]
        y_boot = y[indices]

        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=c_value,
            max_iter=max_iter,
            tol=1e-4,
        )
        model.fit(X_boot, y_boot)
        coefs = model.coef_.ravel()
        coef_history[i, :] = coefs

        mask = np.abs(coefs) > coef_threshold
        selection_counts += mask.astype(int)
        print(f"Iteration {i + 1}/{n_bootstrap}: selected {mask.sum()} encoded features.")

    encoded_stats = []
    for idx, name in enumerate(feature_names):
        encoded_stats.append(
            {
                "encoded_feature": name,
                "raw_feature": encoded_to_raw[name],
                "selection_count": int(selection_counts[idx]),
                "selection_pct": selection_counts[idx] / n_bootstrap,
                "coef_mean": coef_history[:, idx].mean(),
                "coef_std": coef_history[:, idx].std(ddof=0),
            }
        )

    encoded_df = pd.DataFrame(encoded_stats).sort_values(
        by="selection_pct", ascending=False
    )

    raw_stats: Dict[str, Dict[str, float]] = {}
    for raw_feature in sorted(set(encoded_to_raw.values())):
        indices = [i for i, name in enumerate(feature_names) if encoded_to_raw[name] == raw_feature]
        raw_selection = selection_counts[indices] / n_bootstrap
        raw_stats[raw_feature] = {
            "raw_feature": raw_feature,
            "encoded_components": len(indices),
            "mean_selection_pct": float(raw_selection.mean()),
            "max_selection_pct": float(raw_selection.max()),
            "mean_abs_coef": float(np.mean(np.abs(coef_history[:, indices]), axis=1).mean()),
        }

    raw_df = pd.DataFrame(raw_stats.values()).sort_values(
        by="mean_selection_pct", ascending=False
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_path = output_dir / "encoded_feature_stability.csv"
    raw_path = output_dir / "raw_feature_stability.csv"
    encoded_df.to_csv(encoded_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Bootstrap Feature Selection Summary\n")
        handle.write(f"Data file: {data_path}\n")
        handle.write(f"Target column: {target_column}\n")
        handle.write(
            f"Iterations: {n_bootstrap}, sample fraction: {sample_fraction}, C={c_value}\n"
        )
        handle.write(
            "Top 10 raw features by mean selection pct:\n"
        )
        handle.write(raw_df.head(10).to_string(index=False))
        handle.write("\n")

    print(f"Saved encoded-level stats to {encoded_path}")
    print(f"Saved raw-level stats to {raw_path}")
    print(f"Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap L1 feature selection stability analyzer"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../cleaned_data.csv"),
        help="Path to the cleaned dataset CSV (default: ../cleaned_data.csv)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="readmit_label",
        help="Binary target column to predict (default: readmit_label)",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help="Optional CSV/TXT listing features to consider (default: use all columns except target)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bootstrap_results"),
        help="Directory for output CSVs and summary (will be created if missing)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.7,
        help="Fraction of rows to sample (with replacement) per bootstrap iteration (default: 0.7)",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.1,
        help="Inverse regularization strength C for LogisticRegression (default: 0.1)",
    )
    parser.add_argument(
        "--coef-threshold",
        type=float,
        default=1e-6,
        help="Absolute coefficient threshold to treat as selected (default: 1e-6)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for LogisticRegression solver (default: 500)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible bootstraps (default: 42)",
    )

    args = parser.parse_args()

    run_bootstrap(
        data_path=args.data_path,
        target_column=args.target_column,
        feature_file=args.feature_file,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        sample_fraction=args.sample_fraction,
        c_value=args.c_value,
        coef_threshold=args.coef_threshold,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
