"""
Step 1: Load MIMIC-IV data - MEMORY OPTIMIZED VERSION
Processes data chunk by chunk WITHOUT accumulating in memory
Writes directly to parquet files
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path='../config.yaml'):
    """Load configuration"""
    import os
    
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_cohort_with_readmission(admissions_df, patients_df, min_age=18):
    """Create cohort with 30-day readmission labels"""
    print("Creating cohort with readmission labels...")
    
    # Merge with patients
    cohort = admissions_df.merge(patients_df, on='subject_id', how='inner')
    print(f"  Before exclusions: {len(cohort):,} admissions")
    
    # Apply exclusions
    cohort = cohort[cohort['anchor_age'] >= min_age]
    print(f"  After age filter (>={min_age}): {len(cohort):,}")
    
    cohort = cohort[cohort['dischtime'].notna()]
    print(f"  After discharge time filter: {len(cohort):,}")
    
    cohort = cohort[cohort['deathtime'].isna()]
    print(f"  After removing deaths: {len(cohort):,}")
    
    # Calculate length of stay
    cohort['los_hours'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / 3600
    cohort = cohort[cohort['los_hours'] >= 48]
    print(f"  After LOS >= 48h filter: {len(cohort):,}")
    
    # Compute readmission labels
    print("Computing readmission labels...")
    cohort = cohort.sort_values(['subject_id', 'admittime'])
    cohort['next_admittime'] = cohort.groupby('subject_id')['admittime'].shift(-1)
    cohort['days_to_readmit'] = (cohort['next_admittime'] - cohort['dischtime']).dt.days
    
    cohort['readmit_30d'] = ((cohort['days_to_readmit'] <= 30) & (cohort['days_to_readmit'] >= 0)).astype(int)
    cohort['readmit_60d'] = ((cohort['days_to_readmit'] <= 60) & (cohort['days_to_readmit'] >= 0)).astype(int)
    
    print(f"\n  Final cohort size: {len(cohort):,}")
    print(f"  Readmission rate (30d): {cohort['readmit_30d'].mean()*100:.2f}%")
    print(f"  Readmission rate (60d): {cohort['readmit_60d'].mean()*100:.2f}%")
    
    return cohort[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'anchor_age', 
                   'anchor_year', 'anchor_year_group', 'gender', 'readmit_30d', 'readmit_60d']]


def process_events_streaming(csv_path, cohort_hadm_ids, output_path, chunk_size=500000):
    """
    Process large CSV file in streaming mode - never load full file into memory
    """
    file_size_gb = os.path.getsize(csv_path) / (1024**3)
    print(f"\nFile: {os.path.basename(csv_path)} ({file_size_gb:.1f} GB)")
    print(f"Processing in streaming mode (chunk size: {chunk_size:,} rows)")
    
    chunk_num = 0
    total_rows = 0
    kept_rows = 0
    first_chunk = True
    
    try:
        for chunk in pd.read_csv(csv_path, 
                                 chunksize=chunk_size,
                                 on_bad_lines='warn',
                                 low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            
            # Filter to cohort
            chunk_filtered = chunk[chunk['hadm_id'].isin(cohort_hadm_ids)].copy()
            kept_rows += len(chunk_filtered)
            
            # Write to parquet (append mode)
            if len(chunk_filtered) > 0:
                if first_chunk:
                    # First chunk: create new file
                    chunk_filtered.to_parquet(output_path, 
                                             compression='gzip', 
                                             index=False,
                                             engine='fastparquet')
                    first_chunk = False
                else:
                    # Subsequent chunks: append
                    chunk_filtered.to_parquet(output_path, 
                                             compression='gzip', 
                                             index=False,
                                             engine='fastparquet',
                                             append=True)
            
            # Progress every 10 chunks
            if chunk_num % 10 == 0:
                pct = kept_rows/total_rows*100 if total_rows > 0 else 0
                print(f"  Chunk {chunk_num}: {total_rows:,} rows processed, {kept_rows:,} kept ({pct:.1f}%)")
    
    except Exception as e:
        print(f"⚠️  Error at chunk {chunk_num}: {e}")
        print(f"  Continuing with data saved so far...")
    
    print(f"✓ Completed: {total_rows:,} total rows, {kept_rows:,} kept ({kept_rows/total_rows*100:.1f}%)")
    return kept_rows


def main():
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 1: Load Data (MEMORY OPTIMIZED)")
    print("="*80)
    
    # Load config
    config = load_config()
    data_paths = config['data_paths']
    output_dir = config.get('paths', {}).get('output_dir', '../output')
    min_age = config.get('preprocessing', {}).get('min_age', 18)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1.1: Load core tables
    print("\n" + "="*80)
    print("Step 1.1: Loading core MIMIC-IV tables")
    print("="*80)
    
    print("\nLoading patients...")
    try:
        patients = pd.read_csv(data_paths['patients'])
    except TypeError:
        patients = pd.read_csv(data_paths['patients'], engine='c', dtype=str)
        patients['subject_id'] = pd.to_numeric(patients['subject_id'])
        patients['anchor_age'] = pd.to_numeric(patients['anchor_age'])
        patients['anchor_year'] = pd.to_numeric(patients['anchor_year'])
    patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')
    print(f"✓ Loaded {len(patients):,} patients")
    
    print("\nLoading admissions...")
    try:
        admissions = pd.read_csv(data_paths['admissions'])
    except TypeError:
        admissions = pd.read_csv(data_paths['admissions'], engine='c', dtype=str)
        for col in ['subject_id', 'hadm_id']:
            admissions[col] = pd.to_numeric(admissions[col])
    for col in ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']:
        if col in admissions.columns:
            admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
    print(f"✓ Loaded {len(admissions):,} admissions")
    
    # Step 1.2: Create cohort
    print("\n" + "="*80)
    print("Step 1.2: Creating cohort with readmission labels")
    print("="*80)
    
    cohort = create_cohort_with_readmission(admissions, patients, min_age)
    cohort_hadm_ids = set(cohort['hadm_id'].values)
    
    # Save cohort
    cohort = cohort.reset_index(drop=True)
    cohort_path = os.path.join(output_dir, 'cohort.parquet')
    cohort.to_parquet(cohort_path, compression='gzip', index=False, engine='fastparquet')
    print(f"\n✓ Cohort saved to {cohort_path}")
    
    # Step 1.3: Process chartevents in streaming mode
    print("\n" + "="*80)
    print("Step 1.3: Processing chartevents (STREAMING MODE)")
    print("="*80)
    
    if os.path.exists(data_paths['chartevents']):
        chartevents_path = os.path.join(output_dir, 'chartevents_raw.parquet')
        process_events_streaming(
            data_paths['chartevents'], 
            cohort_hadm_ids, 
            chartevents_path,
            chunk_size=500000
        )
        print(f"✓ Saved to {chartevents_path}")
    else:
        print(f"⚠️  Chartevents file not found")
    
    # Step 1.4: Process labevents in streaming mode
    print("\n" + "="*80)
    print("Step 1.4: Processing labevents (STREAMING MODE)")
    print("="*80)
    
    if os.path.exists(data_paths['labevents']):
        labevents_path = os.path.join(output_dir, 'labevents_raw.parquet')
        process_events_streaming(
            data_paths['labevents'],
            cohort_hadm_ids,
            labevents_path,
            chunk_size=500000
        )
        print(f"✓ Saved to {labevents_path}")
    else:
        print(f"⚠️  Labevents file not found")
    
    # Step 1.5: Process prescriptions
    print("\n" + "="*80)
    print("Step 1.5: Processing prescriptions")
    print("="*80)
    
    if os.path.exists(data_paths['prescriptions']):
        print(f"\nLoading prescriptions...")
        prescriptions = pd.read_csv(data_paths['prescriptions'])
        print(f"✓ Loaded {len(prescriptions):,} prescriptions")
        
        prescriptions_cohort = prescriptions[
            prescriptions['hadm_id'].isin(cohort_hadm_ids)
        ].copy()
        print(f"✓ Prescriptions for cohort: {len(prescriptions_cohort):,}")
        
        # Fix data types for parquet compatibility
        for col in prescriptions_cohort.columns:
            if prescriptions_cohort[col].dtype == 'object':
                prescriptions_cohort[col] = prescriptions_cohort[col].astype(str)
        
        prescriptions_path = os.path.join(output_dir, 'prescriptions_raw.parquet')
        prescriptions_cohort.to_parquet(prescriptions_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Saved to {prescriptions_path}")
    else:
        print(f"⚠️  Prescriptions file not found")
    
    print("\n" + "="*80)
    print("✅ STEP 1 COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}")
    print(f"  - cohort.parquet ({len(cohort):,} admissions)")
    print("\nNext step: Run step2_clean_units.py")


if __name__ == "__main__":
    main()
