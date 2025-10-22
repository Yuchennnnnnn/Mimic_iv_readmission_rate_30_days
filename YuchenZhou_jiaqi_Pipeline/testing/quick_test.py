#!/usr/bin/env python3
"""
Quick Test Inference Script - Make predictions using trained models
"""

import sys
import os

print("="*80)
print("  Model Inference Testing - Yuchen Zhou's Pipeline")
print("="*80)
print()

# Check trained models
print("[1/4] Checking trained models...")
models = {
    'Logistic Regression': '../training/artifacts/lr.pkl',
    'Random Forest': '../training/artifacts/rf.pkl',
    'XGBoost': '../training/artifacts/xgb.pkl',
    'LSTM': '../training/artifacts/lstm.pth',
    'Transformer': '../training/artifacts/transformer.pth'
}

available_models = {}
for name, path in models.items():
    if os.path.exists(path):
        available_models[name] = path
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: Not found")

if not available_models:
    print("\n❌ Error: No trained models found")
    print("   Please run training first: cd ../training && python src/train.py --model xgb")
    sys.exit(1)

print(f"\nFound {len(available_models)} trained model(s)")

# Check data files
print("\n[2/4] Checking data files...")
data_path = '../../cleaned_data.csv'

if not os.path.exists(data_path):
    print(f"❌ Error: Data file not found: {data_path}")
    sys.exit(1)

print(f"✓ Data file found: {data_path}")

# Select model
print("\n[3/4] Select a model to test:")
model_list = list(available_models.items())
for i, (name, _) in enumerate(model_list, 1):
    print(f"  {i}. {name}")

choice = input(f"\nPlease select [1-{len(model_list)}] (default: 1): ").strip()

if not choice:
    choice = '1'

try:
    idx = int(choice) - 1
    model_name, model_path = model_list[idx]
except (ValueError, IndexError):
    print("❌ Invalid selection, using default model")
    model_name, model_path = model_list[0]

print(f"\nSelected model: {model_name}")

# Run inference
print("\n[4/4] Running inference...")
print("="*80)

# Select inference method based on model type
if 'pkl' in model_path:  # Traditional ML models
    print(f"\n>>> Using {model_name} for prediction <<<\n")
    
    cmd = f"python src/inference.py --model {model_path} --data {data_path} --output reports/predictions.csv"
    print(f"Command: {cmd}\n")
    
    import subprocess
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ Inference completed successfully!")
        print("="*80)
        print("\n📊 View results:")
        print("  - Predictions: reports/predictions.csv")
        print("  - Evaluation report: reports/evaluation_report.txt")
        print("  - Visualizations: reports/*.png")
    else:
        print("\n❌ Inference failed")
        sys.exit(1)
        
else:  # Deep learning models
    print(f"\n⚠️  {model_name} is a deep learning model that requires special inference code")
    print("   The current inference.py mainly supports traditional ML models (LR, RF, XGBoost)")
    print("\nSuggestion: Use XGBoost or Random Forest for testing")
