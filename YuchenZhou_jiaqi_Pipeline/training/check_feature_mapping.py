#!/usr/bin/env python3
"""
Check LASSO feature to original data column mapping
"""

import pandas as pd
import sys
sys.path.insert(0, 'src')

from feature_selection import load_feature_importance, select_top_features, map_feature_to_column

# Load data
print("Loading data...")
data = pd.read_csv('../../cleaned_data.csv')
print(f"Data shape: {data.shape}")
print(f"Original columns: {len(data.columns)}\n")

# Load feature importance
print("="*80)
print("Loading LASSO Feature Importance")
print("="*80)
feat_imp_df = load_feature_importance('../Feature_Importance_by_Coef.csv')
print(f"Total LASSO features: {len(feat_imp_df)}\n")

# Select top 50
selected_features = select_top_features(feat_imp_df, top_n=50, importance_threshold=0.05)
print(f"Selected LASSO features: {len(selected_features)}\n")

# Map to original columns
print("="*80)
print("Feature Mapping Details")
print("="*80)

mapping = {}
for feat in selected_features:
    col = map_feature_to_column(feat, data.columns.tolist())
    if col:
        if col not in mapping:
            mapping[col] = []
        mapping[col].append(feat)

print(f"\nLASSO Features → Original Columns Mapping:")
print(f"  {len(selected_features)} LASSO features → {len(mapping)} original columns\n")

print("Detailed mapping:")
print("-"*80)
for i, (col, feats) in enumerate(sorted(mapping.items()), 1):
    print(f"\n{i}. Original column: {col}")
    print(f"   Corresponding LASSO features ({len(feats)}):")
    for feat in feats:
        imp = feat_imp_df[feat_imp_df['feature'] == feat]['importance'].values[0]
        print(f"     - {feat:50s} (importance: {imp:.4f})")

# Statistics
print("\n" + "="*80)
print("Statistical Summary")
print("="*80)

one_to_many = {col: feats for col, feats in mapping.items() if len(feats) > 1}
one_to_one = {col: feats for col, feats in mapping.items() if len(feats) == 1}

print(f"\nOne-to-one mapping (original column = LASSO feature): {len(one_to_one)}")
for col in sorted(one_to_one.keys()):
    print(f"  - {col}")

print(f"\nOne-to-many mapping (original column contains multiple LASSO features): {len(one_to_many)}")
for col, feats in sorted(one_to_many.items()):
    print(f"  - {col}: {len(feats)} LASSO features")
    
print("\n" + "="*80)
print("Conclusion")
print("="*80)
print(f"""
✓ LASSO selected {len(selected_features)} features (after One-Hot encoding)
✓ These features map to {len(mapping)} original data columns
✓ {len(one_to_one)} columns are direct matches
✓ {len(one_to_many)} columns contain multiple One-Hot encoded values

This is normal! LASSO was trained on One-Hot encoded data,
while our original data contains categorical columns before encoding.
""")
