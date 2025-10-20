import pandas as pd
from pathlib import Path

def clean_readmission_data_strict(
    input_path="datasets/readmission_features_30d_v1.csv",
    output_path="datasets/cleaned_data.csv",
    col_null_ratio_thresh=0.70,  # Columns with more than 70% missing values will be dropped
):
    print("🧹 Loading dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns")

    # --- Step 1: Drop rows with missing key fields ---
    must_have_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "readmit_label"]
    df = df.dropna(subset=must_have_cols)

    # --- Step 2: Convert time columns and remove invalid time rows ---
    for c in ["admittime", "dischtime"]:
        if df[c].dtype == "object":
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["admittime", "dischtime"])
    df = df[df["dischtime"] >= df["admittime"]]

    # --- Step 3: Drop columns with high missing ratio ---
    null_ratio = df.isna().mean()
    drop_cols = null_ratio[null_ratio > col_null_ratio_thresh].index.tolist()
    if drop_cols:
        print(f"⚠️ Dropping {len(drop_cols)} columns (>70% NaN): {drop_cols}")
        df = df.drop(columns=drop_cols)

    # --- Step 4: Drop any remaining rows that still contain NaN ---
    before_drop = len(df)
    df = df.dropna()
    print(f"✅ Dropped {before_drop - len(df):,} rows containing NaN values")

    # --- Step 5: Save cleaned result ---
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"🎯 Cleaned dataset saved to: {output_path}")
    print(f"🧾 Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    clean_readmission_data_strict()
