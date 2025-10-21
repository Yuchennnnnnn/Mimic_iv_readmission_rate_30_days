#!/usr/bin/env python3
"""
检查LASSO特征到原始数据列的映射
"""

import pandas as pd
import sys
sys.path.insert(0, 'src')

from feature_selection import load_feature_importance, select_top_features, map_feature_to_column

# 加载数据
print("加载数据...")
data = pd.read_csv('../../cleaned_data.csv')
print(f"数据形状: {data.shape}")
print(f"原始列数: {len(data.columns)}\n")

# 加载特征重要性
print("="*80)
print("加载LASSO特征重要性")
print("="*80)
feat_imp_df = load_feature_importance('../Feature_Importance_by_Coef.csv')
print(f"LASSO特征总数: {len(feat_imp_df)}\n")

# 选择top 50
selected_features = select_top_features(feat_imp_df, top_n=50, importance_threshold=0.05)
print(f"选择的LASSO特征数: {len(selected_features)}\n")

# 映射到原始列
print("="*80)
print("特征映射详情")
print("="*80)

mapping = {}
for feat in selected_features:
    col = map_feature_to_column(feat, data.columns.tolist())
    if col:
        if col not in mapping:
            mapping[col] = []
        mapping[col].append(feat)

print(f"\nLASSO特征 → 原始列映射:")
print(f"  {len(selected_features)} 个LASSO特征 → {len(mapping)} 个原始列\n")

print("详细映射:")
print("-"*80)
for i, (col, feats) in enumerate(sorted(mapping.items()), 1):
    print(f"\n{i}. 原始列: {col}")
    print(f"   对应的LASSO特征 ({len(feats)}个):")
    for feat in feats:
        imp = feat_imp_df[feat_imp_df['feature'] == feat]['importance'].values[0]
        print(f"     - {feat:50s} (重要性: {imp:.4f})")

# 统计
print("\n" + "="*80)
print("统计汇总")
print("="*80)

one_to_many = {col: feats for col, feats in mapping.items() if len(feats) > 1}
one_to_one = {col: feats for col, feats in mapping.items() if len(feats) == 1}

print(f"\n一对一映射 (原始列 = LASSO特征): {len(one_to_one)} 个")
for col in sorted(one_to_one.keys()):
    print(f"  - {col}")

print(f"\n一对多映射 (原始列包含多个LASSO特征): {len(one_to_many)} 个")
for col, feats in sorted(one_to_many.items()):
    print(f"  - {col}: {len(feats)} 个LASSO特征")
    
print("\n" + "="*80)
print("结论")
print("="*80)
print(f"""
✓ LASSO选择了 {len(selected_features)} 个特征（One-Hot编码后的）
✓ 这些特征映射到 {len(mapping)} 个原始数据列
✓ 其中 {len(one_to_one)} 列是直接匹配
✓ 其中 {len(one_to_many)} 列包含多个One-Hot编码值

这是正常的！因为LASSO是在One-Hot编码后的数据上训练的，
而我们的原始数据是编码前的categorical columns。
""")
