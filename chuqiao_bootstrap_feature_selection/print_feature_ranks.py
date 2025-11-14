import pandas as pd

path = "chuqiao_bootstrap_feature_selection/bootstrap_results/raw_feature_stability.csv"

df = pd.read_csv(path)
df_sorted = df.sort_values("mean_selection_pct", ascending=False).reset_index(drop=True)

print("=== Full Feature Importance Ranking (Bootstrap Stability) ===")
for i, row in enumerate(df_sorted.itertuples(), start=1):
    print(f"{i:3d}. {row.raw_feature:<35}  Stability={row.mean_selection_pct:.3f}  |  Mean|coef|={row.mean_abs_coef:.4f}")
df_sorted.to_csv("chuqiao_bootstrap_feature_selection/bootstrap_results/feature_importance_ranked.csv", index=False)
print("\n✅ Full ranking saved to bootstrap_outputs/feature_importance_ranked.csv")
