#!/usr/bin/env python3
"""Run only the transformer portion of Yuchen's pipeline for easier debugging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

# Project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = PROJECT_ROOT / "YuchenZhou_jiaqi_Pipeline" / "training"
SRC_DIR = PIPELINE_DIR / "src"

# Ensure we can import Yuchen's modules
sys.path.insert(0, str(SRC_DIR))

import preprocess  # type: ignore
import models  # type: ignore
from dataset import ReadmissionDataset, create_dataloaders, collate_fn  # type: ignore
from utils import load_config, set_seed, get_device, ensure_dir  # type: ignore
from feature_selection import filter_data_by_importance, print_feature_importance_summary  # type: ignore
from train import train_pytorch_model, evaluate_pytorch_model  # type: ignore

# Reuse helpers from the bootstrap runner
from run_bootstrap_training import (
    validate_inputs,
    load_ranking,
    build_feature_file,
    prepare_config,
)


def parse_args() -> argparse.Namespace:
    default_ranking = THIS_DIR.parent / "bootstrap_results" / "feature_importance_ranked.csv"
    default_base_config = PIPELINE_DIR / "config.yaml"
    default_config_out = THIS_DIR / "generated_config.yaml"

    parser = argparse.ArgumentParser(
        description=(
            "Generate a transformer-only config from the bootstrap ranking and "
            "run just that model with full debug visibility."
        )
    )
    parser.add_argument("--top-k", type=int, default=50, help="Number of bootstrap-ranked features to keep.")
    parser.add_argument("--ranking-path", type=Path, default=default_ranking,
                        help="Path to bootstrap feature ranking CSV.")
    parser.add_argument("--base-config", type=Path, default=default_base_config,
                        help="Baseline YAML config from Yuchen's pipeline.")
    parser.add_argument("--config-path", type=Path, default=default_config_out,
                        help="Where to write (or read) the debug config.")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Assume feature file/config already exist and skip regeneration.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print selected bootstrap features before training.")
    return parser.parse_args()


def prepare_environment(args: argparse.Namespace) -> Path:
    """Create feature file + config unless --skip-prep is passed."""
    artifacts_dir = THIS_DIR / "artifacts"
    reports_dir = THIS_DIR / "reports"
    feature_file = artifacts_dir / "bootstrap_top_features.csv"

    if args.skip_prep:
        if not args.config_path.exists():
            raise FileNotFoundError(
                f"--skip-prep was set but config path does not exist: {args.config_path}"
            )
        return args.config_path

    validate_inputs(args.ranking_path, args.base_config)
    ranking_df = load_ranking(args.ranking_path)
    selected_df = build_feature_file(ranking_df, args.top_k, feature_file)

    # Only run transformer in this config
    prepare_config(
        args.base_config,
        args.config_path,
        feature_file,
        args.top_k,
        ["transformer"],
        artifacts_dir,
        reports_dir,
    )

    if args.verbose:
        print(f"Selected top {len(selected_df)} features for transformer debug run:")
        for idx, row in enumerate(selected_df.itertuples(), start=1):
            print(
                f"{idx:2d}. {row.feature} | stability={row.mean_selection_pct:.3f} "
                f"coef={row.mean_abs_coef:.4f}"
            )

    return args.config_path


def run_transformer_only(config_path: Path) -> None:
    config = load_config(str(config_path))
    config['models_to_run'] = ['transformer']

    set_seed(config['seed'], config['deterministic'])
    device = get_device(config['device'])

    data_path = config['data']['input_path']
    df = preprocess.load_data(data_path)

    if config.get('feature_selection', {}).get('enabled', False):
        feature_imp_path = config['data'].get('feature_importance_path')
        if feature_imp_path and Path(feature_imp_path).exists():
            print("\n" + "=" * 60)
            print("FEATURE SELECTION (transformer debug)")
            print("=" * 60)
            print_feature_importance_summary(feature_imp_path, top_n=min(20, config['feature_selection'].get('top_n', 20)))

            df, selected_features = filter_data_by_importance(
                df,
                feature_imp_path,
                top_n=config['feature_selection'].get('top_n'),
                importance_threshold=config['feature_selection'].get('importance_threshold'),
                keep_id_cols=config['columns']['id_cols'],
                keep_label_col=config['columns']['label']
            )
            print(f"Using {len(selected_features)} features after bootstrap filtering.")
        else:
            print("Feature importance file not found; using all features.")

    categorical_cols, numeric_cols = preprocess.detect_column_types(df, config)
    available_cols = set(df.columns)
    categorical_cols = [col for col in categorical_cols if col in available_cols]
    numeric_cols = [col for col in numeric_cols if col in available_cols]
    print(f"\nDetected {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns after selection.")

    df = preprocess.handle_missing_values(df, categorical_cols, numeric_cols, config)

    id_cols = config['columns']['id_cols']
    label_col = config['columns']['label']
    time_cols = config['columns'].get('time_cols', [])
    drop_cols = id_cols + [label_col] + time_cols

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[label_col].values
    ids = df[id_cols[1]].values if len(id_cols) > 1 else df[id_cols[0]].values

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state'],
        stratify=y if config['split']['stratify'] else None
    )

    output_dir = config['data']['output_dir']
    ensure_dir(output_dir)

    encoder_info = preprocess.build_encoders_for_deep(
        X_train, categorical_cols, numeric_cols, config, output_dir, model_name='transformer'
    )

    cat_train, cont_train = preprocess.transform_for_deep(
        X_train,
        encoder_info['cat_to_id_maps'],
        encoder_info['scaler'],
        encoder_info['categorical_cols'],
        encoder_info['numeric_cols']
    )
    cat_test, cont_test = preprocess.transform_for_deep(
        X_test,
        encoder_info['cat_to_id_maps'],
        encoder_info['scaler'],
        encoder_info['categorical_cols'],
        encoder_info['numeric_cols']
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.15,
        random_state=config['seed'],
        stratify=y_train
    )

    train_dataset = ReadmissionDataset(
        {k: v[train_idx] for k, v in cat_train.items()},
        cont_train[train_idx],
        y_train[train_idx]
    )
    val_dataset = ReadmissionDataset(
        {k: v[val_idx] for k, v in cat_train.items()},
        cont_train[val_idx],
        y_train[val_idx]
    )
    test_dataset = ReadmissionDataset(cat_test, cont_test, y_test)

    batch_size = config['hyperparameters']['transformer']['batch_size']
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    continuous_dim = cont_train.shape[1]
    model = models.create_transformer_model(
        config,
        encoder_info['vocab_sizes'],
        encoder_info['embedding_dims'],
        continuous_dim,
        use_tab_transformer=True
    )

    trained_model, history = train_pytorch_model(
        model,
        train_loader,
        val_loader,
        'transformer',
        config,
        output_dir,
        device
    )

    metrics = evaluate_pytorch_model(
        trained_model,
        test_loader,
        'transformer',
        config,
        device
    )

    print("Transformer-only run complete. Metrics: ", metrics)


def main() -> None:
    args = parse_args()
    config_path = prepare_environment(args)
    run_transformer_only(config_path)


if __name__ == "__main__":
    main()
