"""
Step 2: Clean and Standardize Units
Processes raw event data from BigQuery and standardizes units
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import yaml
from tqdm import tqdm
from utils import (
    load_config, standardize_units, ITEMID_TO_FEATURE,
    apply_outlier_bounds
)


def map_itemid_to_feature(df, itemid_map):
    """Map itemid to feature names"""
    df = df.copy()
    df['feature_name'] = df['itemid'].map(itemid_map)
    # Drop rows with unmapped itemids
    df = df[df['feature_name'].notna()]
    return df


def clean_chartevents(df):
    """Clean chart events data"""
    print("Cleaning chart events...")
    
    # Map itemids to feature names
    df = map_itemid_to_feature(df, ITEMID_TO_FEATURE)
    print(f"  After mapping: {len(df):,} rows")
    
    # Remove duplicates (same hadm_id, charttime, feature_name)
    df = df.drop_duplicates(subset=['hadm_id', 'charttime', 'feature_name'])
    print(f"  After deduplication: {len(df):,} rows")
    
    # Standardize units by feature
    for feature_name in df['feature_name'].unique():
        mask = df['feature_name'] == feature_name
        df.loc[mask] = standardize_units(df[mask], feature_name)
    
    # Keep only necessary columns (rename valueuom to unit for consistency)
    df = df[['subject_id', 'hadm_id', 'charttime', 'feature_name', 'valuenum', 'valueuom']]
    df = df.rename(columns={'valueuom': 'unit'})
    
    return df


def clean_labevents(df):
    """Clean lab events data"""
    print("Cleaning lab events...")
    
    # Map itemids to feature names
    df = map_itemid_to_feature(df, ITEMID_TO_FEATURE)
    print(f"  After mapping: {len(df):,} rows")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['hadm_id', 'charttime', 'feature_name'])
    print(f"  After deduplication: {len(df):,} rows")
    
    # Keep only necessary columns (rename valueuom to unit for consistency)
    df = df[['subject_id', 'hadm_id', 'charttime', 'feature_name', 'valuenum', 'valueuom']]
    df = df.rename(columns={'valueuom': 'unit'})
    
    return df


def clean_prescriptions(df):
    """Clean prescriptions data"""
    print("Cleaning prescriptions...")
    
    if df.empty:
        print("  No data to clean")
        return df
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['hadm_id', 'starttime', 'drug'])
    print(f"  After deduplication: {len(df):,} rows")
    
    # Keep only necessary columns
    df = df[['subject_id', 'hadm_id', 'starttime', 'stoptime', 
             'drug', 'medication_category']]
    
    return df


def main():
    """Main execution"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 2: Clean Units")
    print("="*80)
    
    # Load config
    config = load_config()
    output_dir = config['paths']['output_dir']
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load raw data from Step 1
    print("\nLoading raw data...")
    
    cohort = pd.read_parquet(os.path.join(output_dir, 'cohort.parquet'), engine='fastparquet')
    
    # Only read necessary columns to save memory
    chart_columns = ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom']
    lab_columns = ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom']
    
    print("  Loading chartevents (large file, please wait)...")
    chartevents = pd.read_parquet(
        os.path.join(output_dir, 'chartevents_raw.parquet'), 
        engine='fastparquet',
        columns=chart_columns
    )
    
    print("  Loading labevents (large file, please wait)...")
    labevents = pd.read_parquet(
        os.path.join(output_dir, 'labevents_raw.parquet'), 
        engine='fastparquet',
        columns=lab_columns
    )
    
    # Skip prescriptions if not available (it's optional)
    prescriptions_path = os.path.join(output_dir, 'prescriptions_raw.parquet')
    if os.path.exists(prescriptions_path):
        prescriptions = pd.read_parquet(prescriptions_path, engine='fastparquet')
        print(f"✓ Prescriptions: {len(prescriptions):,} rows")
    else:
        print("⚠ Prescriptions file not found - skipping (optional)")
        prescriptions = pd.DataFrame()
    
    print(f"✓ Cohort: {len(cohort):,} admissions")
    print(f"✓ Chart events: {len(chartevents):,} rows")
    print(f"✓ Lab events: {len(labevents):,} rows")
    print(f"✓ Prescriptions: {len(prescriptions):,} rows")
    
    # Clean each dataset
    print("\n" + "-"*80)
    chartevents_clean = clean_chartevents(chartevents)
    
    print("\n" + "-"*80)
    labevents_clean = clean_labevents(labevents)
    
    # Only process prescriptions if data exists
    if not prescriptions.empty:
        print("\n" + "-"*80)
        prescriptions_clean = clean_prescriptions(prescriptions)
    else:
        prescriptions_clean = pd.DataFrame()
    
    # Save cleaned data
    print("\n" + "="*80)
    print("Saving cleaned data...")
    
    chartevents_path = os.path.join(output_dir, 'chartevents_clean.parquet')
    chartevents_clean.to_parquet(chartevents_path, compression='gzip', index=False, engine='fastparquet')
    print(f"✓ Chart events: {chartevents_path}")
    
    labevents_path = os.path.join(output_dir, 'labevents_clean.parquet')
    labevents_clean.to_parquet(labevents_path, compression='gzip', index=False, engine='fastparquet')
    print(f"✓ Lab events: {labevents_path}")
    
    if not prescriptions_clean.empty:
        prescriptions_path = os.path.join(output_dir, 'prescriptions_clean.parquet')
        prescriptions_clean.to_parquet(prescriptions_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Prescriptions: {prescriptions_path}")
    else:
        print("⚠ Prescriptions: skipped (no data)")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Chart events:   {len(chartevents):>10,} → {len(chartevents_clean):>10,} rows")
    print(f"Lab events:     {len(labevents):>10,} → {len(labevents_clean):>10,} rows")
    print(f"Prescriptions:  {len(prescriptions):>10,} → {len(prescriptions_clean):>10,} rows")
    
    print("\n✓ Step 2 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
