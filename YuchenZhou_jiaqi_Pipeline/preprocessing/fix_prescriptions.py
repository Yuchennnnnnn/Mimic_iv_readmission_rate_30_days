"""
Quick fix: Reload and resave prescriptions with proper data types
"""
import pandas as pd
import os

print("Loading prescriptions...")
prescriptions = pd.read_csv('/Users/yuchenzhou/Documents/duke/compsci526/final_proj/mimic-iv-3.1/hosp/prescriptions.csv', low_memory=False)
print(f"✓ Loaded {len(prescriptions):,} prescriptions")

print("\nLoading cohort...")
cohort = pd.read_parquet('/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/YuchenZhou_jiaqi_Pipeline/output/cohort.parquet', engine='fastparquet')
cohort_hadm_ids = set(cohort['hadm_id'].values)
print(f"✓ Cohort: {len(cohort):,} admissions")

print("\nFiltering prescriptions to cohort...")
prescriptions_cohort = prescriptions[prescriptions['hadm_id'].isin(cohort_hadm_ids)].copy()
print(f"✓ Prescriptions for cohort: {len(prescriptions_cohort):,}")

print("\nFixing data types...")
for col in prescriptions_cohort.columns:
    if prescriptions_cohort[col].dtype == 'object':
        prescriptions_cohort[col] = prescriptions_cohort[col].astype(str)
print("✓ Data types fixed")

print("\nSaving to parquet...")
output_path = '/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/YuchenZhou_jiaqi_Pipeline/output/prescriptions_raw.parquet'
prescriptions_cohort.to_parquet(output_path, compression='gzip', index=False, engine='fastparquet')
print(f"✓ Saved to {output_path}")

# Check file size
size_mb = os.path.getsize(output_path) / (1024**2)
print(f"\n✅ Complete! File size: {size_mb:.1f} MB")
