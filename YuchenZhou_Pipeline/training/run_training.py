#!/usr/bin/env python3
"""
Simple training test script - with feature selection
"""

import os
import sys

print("="*80)
print("  Yuchen Zhou's 30-Day Readmission Prediction Pipeline")
print("  Training with Pre-computed Feature Importance")
print("="*80)
print()

# Check environment
print("[1/5] Checking environment...")
if not os.path.exists('src/train.py'):
    print("❌ Error: Please run this script from YuchenZhou_Pipeline/training directory")
    sys.exit(1)

print("✓ Directory correct")

# Check data files
print("\n[2/5] Checking data files...")
config_path = 'config.yaml'

import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = config['data']['input_path']
feat_imp_path = config['data']['feature_importance_path']

if not os.path.exists(data_path):
    print(f"❌ Error: Data file does not exist: {data_path}")
    print("   Please update input_path in config.yaml")
    sys.exit(1)

print(f"✓ Data file found: {data_path}")

if not os.path.exists(feat_imp_path):
    print(f"⚠️  Warning: Feature importance file does not exist: {feat_imp_path}")
    print("   Will use all features for training")
else:
    print(f"✓ Feature importance file found: {feat_imp_path}")

# Check dependencies
print("\n[3/5] Checking Python dependencies...")
required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib', 'seaborn', 'yaml', 'joblib']
missing_packages = []

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'yaml':
            __import__('yaml')
        else:
            __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"❌ Missing packages: {', '.join(missing_packages)}")
    print("\n   Please install:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("✓ All dependencies installed")

# Display configuration
print("\n[4/5] Training configuration:")
print(f"   Data path: {data_path}")
print(f"   Feature selection: {'Enabled' if config.get('feature_selection', {}).get('enabled') else 'Disabled'}")
if config.get('feature_selection', {}).get('enabled'):
    top_n = config['feature_selection'].get('top_n', 'All')
    threshold = config['feature_selection'].get('importance_threshold', 'None')
    print(f"   - Top N features: {top_n}")
    print(f"   - Importance threshold: {threshold}")
print(f"   Models to train: {', '.join(config.get('models_to_run', []))}")

# Ask user
print("\n[5/5] Preparing for training...")
print("\nSelect training option:")
print("  1. Quick test (Logistic Regression only, ~2 minutes)")
print("  2. Train traditional models (LR + RF + XGBoost, ~10 minutes)")
print("  3. Train all models (including deep learning, ~1 hour)")
print("  4. Exit")

choice = input("\nPlease select [1-4]: ").strip()

if choice == '1':
    model_choice = 'logistic'
    print("\nStarting quick test...")
elif choice == '2':
    model_choice = 'logistic,rf,xgb'
    print("\nStarting training for traditional models...")
elif choice == '3':
    model_choice = 'all'
    print("\nStarting training for all models...")
elif choice == '4':
    print("Exiting")
    sys.exit(0)
else:
    print("Invalid selection, using default: quick test")
    model_choice = 'logistic'

# Run training
print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")

import subprocess

if model_choice == 'all':
    cmd = ['python', 'src/train.py', '--model', 'all', '--config', config_path]
else:
    # For multiple models, train one by one
    models = model_choice.split(',')
    for model in models:
        print(f"\n>>> Training {model.upper()} <<<\n")
        cmd = ['python', 'src/train.py', '--model', model.strip(), '--config', config_path]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ {model} training failed")
            continue
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    sys.exit(0)

# For 'all', run directly
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ Training completed successfully!")
    print("="*80)
    print("\nView results:")
    print("  - Metrics: reports/metrics.csv")
    print("  - Charts: reports/model_comparison.png")
    print("  - Models: artifacts/")
else:
    print("\n" + "="*80)
    print("❌ Error occurred during training")
    print("="*80)
    sys.exit(1)
