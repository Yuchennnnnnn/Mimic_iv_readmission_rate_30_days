#!/usr/bin/env python3
"""
Generate Training and Testing Data Summary Report
"""

import pandas as pd
import os

print("="*80)
print("  Yuchen Zhou's 30-Day Readmission Prediction Pipeline")
print("  Training & Testing Data Summary Report")
print("="*80)
print()

# ============================================================================
# 1. Dataset Basic Information
# ============================================================================
print("📊 Dataset Basic Information")
print("-"*80)

# Read original data
data = pd.read_csv('../cleaned_data.csv')
print(f"Original dataset: {len(data):,} samples")
print(f"Number of features: {data.shape[1]} columns")

# Calculate label distribution
if 'readmit_label' in data.columns:
    readmit_count = data['readmit_label'].sum()
    readmit_rate = data['readmit_label'].mean() * 100
    print(f"Readmission label: {readmit_count:,} cases ({readmit_rate:.2f}%)")
    print(f"No readmission: {len(data) - readmit_count:,} cases ({100-readmit_rate:.2f}%)")

print()

# ============================================================================
# 2. Training data (in training/reports/)
# ============================================================================
print("="*80)
print("🎯 TRAINING Results Summary (20% test set)")
print("="*80)
print()

# Read metrics
os.chdir('training/reports')
metrics = pd.read_csv('metrics.csv')

# Remove duplicates and keep only the last run
metrics = metrics.drop_duplicates(subset=['model'], keep='last')

print("Training configuration:")
print(f"  - Training set: 164,784 samples (80%)")
print(f"  - Test set: 41,196 samples (20%)")
print(f"  - Number of features: 18 (mapped from 48 LASSO features)")
print(f"  - Trained models: {len(metrics)}")
print()

# Performance comparison table
print("Model Performance Comparison (on 20% test set):")
print("-"*80)
print(f"{'Model':<15} {'ROC-AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-"*80)

for _, row in metrics.iterrows():
    print(f"{row['model']:<15} {row['roc_auc']:<10.4f} {row['accuracy']:<10.4f} "
          f"{row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f}")

print("-"*80)
print()

# Find best models
best_auc = metrics.loc[metrics['roc_auc'].idxmax()]
best_f1 = metrics.loc[metrics['f1'].idxmax()]
best_recall = metrics.loc[metrics['recall'].idxmax()]

print("🏆 Best Models:")
print(f"  - Highest ROC-AUC: {best_auc['model']} ({best_auc['roc_auc']:.4f})")
print(f"  - Highest F1-Score: {best_f1['model']} ({best_f1['f1']:.4f})")
print(f"  - Highest Recall: {best_recall['model']} ({best_recall['recall']:.4f})")
print()

# Confusion matrix details
print("Confusion Matrix Details (test set 41,196 samples):")
print("-"*80)
for _, row in metrics.iterrows():
    print(f"\n{row['model']}:")
    tn, fp, fn, tp = row['true_negatives'], row['false_positives'], row['false_negatives'], row['true_positives']
    print(f"  True Negatives (TN): {tn:>6,}  |  False Positives (FP): {fp:>6,}")
    print(f"  False Negatives (FN): {fn:>6,}  |  True Positives (TP): {tp:>6,}")
    print(f"  Specificity: {row['specificity']:.2%}  |  Sensitivity: {row['sensitivity']:.2%}")

print()

# Prediction file statistics
print("Prediction files generated during training (test set):")
print("-"*80)
prediction_files = [f for f in os.listdir('.') if f.startswith('predictions_') and f.endswith('.csv')]

for pred_file in sorted(prediction_files):
    model_name = pred_file.replace('predictions_', '').replace('.csv', '').upper()
    df = pd.read_csv(pred_file)
    print(f"  {model_name}: {len(df):,} predictions")

print()

# ============================================================================
# 3. Testing data (in testing/reports/)
# ============================================================================
os.chdir('../../testing/reports')

print("="*80)
print("🧪 TESTING/INFERENCE Results Summary (full dataset)")
print("="*80)
print()

print("Inference configuration:")
print(f"  - Inference data: Full dataset (205,980 samples)")
print(f"  - Purpose: Generate risk predictions for all samples")
print()

# Check testing prediction files
testing_pred_files = [f for f in os.listdir('.') if f.startswith('predictions_') and f.endswith('.csv')]

if testing_pred_files:
    print("Completed inference tasks:")
    print("-"*80)
    
    for pred_file in sorted(testing_pred_files):
        model_name = pred_file.replace('predictions_', '').replace('.csv', '').upper()
        df = pd.read_csv(pred_file)
        
        total = len(df)
        predicted_positive = df['predicted_label'].sum()
        predicted_ratio = predicted_positive / total * 100
        avg_prob = df['predicted_probability'].mean()
        
        print(f"\n{model_name}:")
        print(f"  Total samples: {total:,}")
        print(f"  Predicted readmission: {predicted_positive:,} ({predicted_ratio:.1f}%)")
        print(f"  Average prediction probability: {avg_prob:.3f}")
        print(f"  Probability range: [{df['predicted_probability'].min():.3f}, {df['predicted_probability'].max():.3f}]")
else:
    print("⚠️  No inference tasks have been run yet")
    print("   Run: cd testing && ./run_all_inference.sh")

print()

# ============================================================================
# 4. File Location Summary
# ============================================================================
os.chdir('../..')
print("="*80)
print("📁 Generated File Locations")
print("="*80)
print()

print("Training Results (training/reports/):")
print("  - metrics.csv - Performance metrics for all models")
print("  - predictions_*.csv - Test set predictions (41,196 samples)")
print("  - roc_curve_*.png - ROC curves")
print("  - confusion_matrix_*.png - Confusion matrices")
print("  - feature_importance_*.png - Feature importance plots")
print()

print("Testing Results (testing/reports/):")
print("  - predictions_*.csv - Full dataset predictions (205,980 samples)")
print()

print("Trained Models (training/artifacts/):")
print("  - lr.pkl, rf.pkl, xgb.pkl - Traditional ML models")
print("  - lstm.pt, transformer.pt - Deep learning models")
print("  - *_preprocessor.joblib - Corresponding preprocessors")
print()

# ============================================================================
# 5. Key Differences Explained
# ============================================================================
print("="*80)
print("📖 Training vs Testing Key Differences")
print("="*80)
print()

print("🎯 TRAINING (training/reports/):")
print("  Data: 20% test set (41,196 samples)")
print("  Purpose: Evaluate model performance, calculate metrics")
print("  Contains: True labels + predictions")
print("  Output: ROC-AUC, Precision, Recall, F1 and other metrics")
print("  Files: predictions_*.csv includes true label comparison")
print()

print("🧪 TESTING/INFERENCE (testing/reports/):")
print("  Data: Full dataset (205,980 samples)")
print("  Purpose: Generate risk predictions for all patients")
print("  Contains: Prediction probabilities + predicted labels")
print("  Output: Readmission risk score for each patient")
print("  Files: predictions_*.csv for practical application")
print()

print("="*80)
print("✅ Report generated successfully!")
print("="*80)
