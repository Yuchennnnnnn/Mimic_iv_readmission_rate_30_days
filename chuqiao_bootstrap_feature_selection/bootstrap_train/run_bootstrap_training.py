#!/usr/bin/env python3
"""Train models with Yuchen's pipeline using bootstrap-ranked features."""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    default_ranking = base_dir.parent / "bootstrap_results" / "feature_importance_ranked.csv"
    default_base_config = project_root / "YuchenZhou_jiaqi_Pipeline" / "training" / "config.yaml"

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a feature-importance file from bootstrap rankings and "
            "launch Yuchen's training pipeline with the top-K features."
        )
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top features (by mean_selection_pct) to keep."
    )
    parser.add_argument(
        "--ranking-path",
        type=Path,
        default=default_ranking,
        help="Path to bootstrap feature importance ranking CSV."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=default_base_config,
        help="Path to the baseline YAML config from Yuchen's pipeline."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logistic,rf,xgb,transformer",
        help="Comma-separated list of models to train when using --model all."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Exit after generating the feature list and config without training."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the selected feature list before training."
    )
    return parser.parse_args()


def validate_inputs(ranking_path: Path, base_config: Path) -> None:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")


def load_ranking(ranking_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ranking_path)
    required_cols = {
        "raw_feature",
        "mean_selection_pct",
        "mean_abs_coef",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Ranking file is missing required columns: {', '.join(sorted(missing))}"
        )
    df = df.sort_values(
        ["mean_selection_pct", "mean_abs_coef"], ascending=[False, False]
    ).reset_index(drop=True)
    return df


def build_feature_file(df: pd.DataFrame, top_k: int, output_path: Path) -> pd.DataFrame:
    selected = df.head(top_k).copy()
    selected = selected.rename(columns={"raw_feature": "feature"})
    selected["importance"] = selected["mean_selection_pct"]
    selected["coef"] = selected["mean_abs_coef"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.loc[:, ["feature", "coef", "importance"]].to_csv(output_path, index=False)
    return selected


def prepare_config(
    base_config_path: Path,
    dest_config_path: Path,
    feature_file: Path,
    top_k: int,
    models: list[str],
    artifacts_dir: Path,
    reports_dir: Path,
) -> None:
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("feature_selection", {})
    config["feature_selection"]["enabled"] = True
    config["feature_selection"]["top_n"] = top_k
    config["feature_selection"]["importance_threshold"] = None

    config.setdefault("data", {})
    config["data"]["feature_importance_path"] = str(feature_file)
    config["data"]["output_dir"] = str(artifacts_dir)
    config["data"]["reports_dir"] = str(reports_dir)

    config["models_to_run"] = models

    dest_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_training(training_dir: Path, config_path: Path) -> None:
    cmd = [sys.executable, "src/train.py", "--model", "all", "--config", str(config_path)]
    subprocess.run(cmd, cwd=training_dir, check=True)


def main() -> None:
    args = parse_args()
    ranking_path = args.ranking_path.resolve()
    base_config = args.base_config.resolve()

    validate_inputs(ranking_path, base_config)

    bootstrap_dir = Path(__file__).resolve().parent
    training_dir = base_config.parent
    artifacts_dir = bootstrap_dir / "artifacts"
    reports_dir = bootstrap_dir / "reports"
    feature_file = bootstrap_dir / "artifacts" / "bootstrap_top_features.csv"
    dest_config = bootstrap_dir / "generated_config.yaml"

    ranking_df = load_ranking(ranking_path)
    selected_df = build_feature_file(ranking_df, args.top_k, feature_file)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("At least one model must be specified via --models")

    prepare_config(
        base_config,
        dest_config,
        feature_file,
        args.top_k,
        models,
        artifacts_dir,
        reports_dir,
    )

    if args.verbose:
        print(f"Selected top {len(selected_df)} features:")
        for idx, row in enumerate(selected_df.itertuples(), start=1):
            print(
                f"{idx:2d}. {row.feature} | stability={row.mean_selection_pct:.3f} "
                f"coef={row.mean_abs_coef:.4f}"
            )

    if args.prepare_only:
        print("Preparation complete. Skipping training per --prepare-only flag.")
        print(f"Feature file: {feature_file}")
        print(f"Config file: {dest_config}")
        return

    print("Launching training with Yuchen's pipeline...")
    run_training(training_dir, dest_config)


if __name__ == "__main__":
    main()
