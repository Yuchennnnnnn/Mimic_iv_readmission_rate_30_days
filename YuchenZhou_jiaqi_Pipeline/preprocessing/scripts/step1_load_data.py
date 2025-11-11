"""
Step 1: Load MIMIC-IV data from local CSV files and create cohort
Processes local CSV files instead of BigQuery
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path='../config.yaml'):
    """Load configuration"""
    import os
    
    # If running from scripts directory, go up one level
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_cohort_with_readmission(admissions_df, patients_df, min_age=18):
    """
    Create cohort with 30-day readmission labels
    
    Args:
        admissions_df: Admissions DataFrame
        patients_df: Patients DataFrame
        min_age: Minimum age for inclusion
    
    Returns:
        Cohort DataFrame with readmission labels
    """
    print("Creating cohort with readmission labels...")
    
    # Merge with patients to get demographics
    cohort = admissions_df.merge(
        patients_df[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod']],
        on='subject_id',
        how='inner'
    )
    
    # Calculate age at admission
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    cohort['age_at_admission'] = cohort['anchor_age'] + \
        (cohort['admittime'].dt.year - cohort['anchor_year'])
    
    # Exclusion criteria
    print(f"  Before exclusions: {len(cohort):,} admissions")
    
    # Must be adult
    cohort = cohort[cohort['age_at_admission'] >= min_age]
    print(f"  After age filter (>={min_age}): {len(cohort):,}")
    
    # Must have valid discharge time
    cohort = cohort[cohort['dischtime'].notna()]
    print(f"  After discharge time filter: {len(cohort):,}")
    
    # Must not have died in hospital
    cohort = cohort[cohort['hospital_expire_flag'] == 0]
    print(f"  After removing deaths: {len(cohort):,}")
    
    # Must have at least 48 hours stay
    cohort['los_hours'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / 3600
    cohort = cohort[cohort['los_hours'] >= 48]
    print(f"  After LOS >= 48h filter: {len(cohort):,}")
    
    # Sort by subject and admission time
    cohort = cohort.sort_values(['subject_id', 'admittime']).reset_index(drop=True)
    
    # Compute next admission for each patient
    print("Computing readmission labels...")
    cohort['next_admittime'] = cohort.groupby('subject_id')['admittime'].shift(-1)
    cohort['next_hadm_id'] = cohort.groupby('subject_id')['hadm_id'].shift(-1)
    
    # Calculate days to readmission
    cohort['days_to_readmit'] = (
        (cohort['next_admittime'] - cohort['dischtime']).dt.total_seconds() / 86400
    )
    
    # 30-day readmission label
    cohort['readmit_30d'] = (
        (cohort['days_to_readmit'] >= 1) & 
        (cohort['days_to_readmit'] <= 30)
    ).astype(int)
    
    # 60-day readmission label (optional)
    cohort['readmit_60d'] = (
        (cohort['days_to_readmit'] >= 1) & 
        (cohort['days_to_readmit'] <= 60)
    ).astype(int)
    
    # Calculate length of stay in days
    cohort['los_days'] = (cohort['dischtime'] - cohort['admittime']).dt.days
    
    print(f"\n  Final cohort size: {len(cohort):,}")
    print(f"  Readmission rate (30d): {cohort['readmit_30d'].mean()*100:.2f}%")
    print(f"  Readmission rate (60d): {cohort['readmit_60d'].mean()*100:.2f}%")
    
    return cohort


def extract_events_for_cohort(events_df, cohort_df, time_col='charttime', window_hours=48):
    """
    Extract events for cohort admissions within time window
    
    Args:
        events_df: Events DataFrame (chartevents, labevents, etc.)
        cohort_df: Cohort DataFrame with hadm_id and admittime
        time_col: Name of timestamp column
        window_hours: Time window in hours
    
    Returns:
        Filtered events DataFrame
    """
    print(f"Extracting events for {len(cohort_df):,} admissions...")
    
    # Get cohort hadm_ids and admittimes
    cohort_lookup = cohort_df[['hadm_id', 'admittime']].set_index('hadm_id')['admittime'].to_dict()
    
    # Filter to cohort admissions
    events_cohort = events_df[events_df['hadm_id'].isin(cohort_lookup.keys())].copy()
    print(f"  Events for cohort: {len(events_cohort):,}")
    
    # Convert time column
    events_cohort[time_col] = pd.to_datetime(events_cohort[time_col])
    
    # Add admittime and compute hours since admission
    events_cohort['admittime'] = events_cohort['hadm_id'].map(cohort_lookup)
    events_cohort['hours_since_admit'] = (
        (events_cohort[time_col] - events_cohort['admittime']).dt.total_seconds() / 3600
    )
    
    # Filter to first N hours
    events_cohort = events_cohort[
        (events_cohort['hours_since_admit'] >= 0) & 
        (events_cohort['hours_since_admit'] < window_hours)
    ]
    print(f"  After time window filter (0-{window_hours}h): {len(events_cohort):,}")
    
    return events_cohort


def main():
    """Main execution function"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 1: Load Data and Create Cohort")
    print("="*80)
    
    # Load configuration
    config = load_config()
    data_paths = config['data_paths']
    output_dir = config['paths']['output_dir']
    window_hours = config['preprocessing']['time_window_hours']
    min_age = config['preprocessing']['min_age']
    
    print(f"\nOutput directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1.1: Load core tables
    print("\n" + "="*80)
    print("Step 1.1: Loading core MIMIC-IV tables")
    print("="*80)
    
    print("\nLoading patients...")
    try:
        patients = pd.read_csv(data_paths['patients'])
    except TypeError:
        # Workaround for pandas/numpy compatibility issue
        import csv
        patients = pd.read_csv(data_paths['patients'], engine='c', dtype=str)
        # Convert numeric columns
        patients['subject_id'] = pd.to_numeric(patients['subject_id'])
        patients['anchor_age'] = pd.to_numeric(patients['anchor_age'])
        patients['anchor_year'] = pd.to_numeric(patients['anchor_year'])
    patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')
    print(f"✓ Loaded {len(patients):,} patients")
    
    print("\nLoading admissions...")
    try:
        admissions = pd.read_csv(data_paths['admissions'])
    except TypeError:
        # Workaround for pandas/numpy compatibility issue
        admissions = pd.read_csv(data_paths['admissions'], engine='c', dtype=str)
        # Convert numeric columns
        for col in ['subject_id', 'hadm_id']:
            admissions[col] = pd.to_numeric(admissions[col])
    # Convert datetime columns
    for col in ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']:
        if col in admissions.columns:
            admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
    print(f"✓ Loaded {len(admissions):,} admissions")
    
    # Step 1.2: Create cohort with readmission labels
    print("\n" + "="*80)
    print("Step 1.2: Creating cohort with readmission labels")
    print("="*80)
    
    cohort = create_cohort_with_readmission(admissions, patients, min_age)
    
    # Reset index to ensure clean dataframe
    cohort = cohort.reset_index(drop=True)
    
    # Save cohort (use fastparquet engine to avoid pyarrow issues)
    cohort_path = os.path.join(output_dir, 'cohort.parquet')
    try:
        cohort.to_parquet(cohort_path, compression='gzip', index=False, engine='fastparquet')
    except:
        # Fallback to pyarrow with reset types
        cohort = cohort.copy()
        for col in cohort.select_dtypes(include=['int']).columns:
            cohort[col] = cohort[col].astype('Int64')
        cohort.to_parquet(cohort_path, compression='gzip', index=False)
    print(f"\n✓ Cohort saved to {cohort_path}")
    
    # Step 1.3: Load and extract chart events
    print("\n" + "="*80)
    print("Step 1.3: Loading and extracting chart events")
    print("="*80)
    
    if os.path.exists(data_paths['chartevents']):
        print(f"\nLoading chartevents from {data_paths['chartevents']}...")
        file_size_gb = os.path.getsize(data_paths['chartevents']) / (1024**3)
        print(f"File size: {file_size_gb:.1f} GB")
        print("⚠️  This is a VERY large file. Processing will take 15-30 minutes...")
        print("💡 TIP: You can monitor progress with: wc -l on the file")
        
        # Load in chunks to save memory
        chunk_size = 500000  # Smaller chunks for 39GB file
        chartevents_list = []
        cohort_hadm_ids = set(cohort['hadm_id'].values)  # Use set for faster lookup
        
        chunk_num = 0
        total_rows = 0
        kept_rows = 0
        
        print("\nProcessing chunks (500K rows each)...")
        print("⚠️  Using error-tolerant mode (will skip malformed rows)")
        
        for chunk in pd.read_csv(data_paths['chartevents'], 
                                 chunksize=chunk_size,
                                 on_bad_lines='warn',  # Warn but continue
                                 engine='c',  # Use fast C engine
                                 low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            
            # Filter to cohort immediately to save memory
            chunk_cohort = chunk[chunk['hadm_id'].isin(cohort_hadm_ids)]
            kept_rows += len(chunk_cohort)
            
            if len(chunk_cohort) > 0:
                chartevents_list.append(chunk_cohort)
            
            # Print progress every 10 chunks
            if chunk_num % 10 == 0:
                print(f"  Processed chunk {chunk_num}: {total_rows:,} rows total, {kept_rows:,} kept ({kept_rows/total_rows*100:.1f}%)")
        
        print(f"\n✓ Finished loading: {total_rows:,} total rows, {kept_rows:,} kept")
        
        chartevents = pd.concat(chartevents_list, ignore_index=True)
        print(f"✓ Concatenated {len(chartevents):,} chart events for cohort")
        
        # Extract events in time window
        chartevents_cohort = extract_events_for_cohort(
            chartevents, cohort, time_col='charttime', window_hours=window_hours
        )
        
        # Save
        chartevents_path = os.path.join(output_dir, 'chartevents_raw.parquet')
        chartevents_cohort.to_parquet(chartevents_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Saved to {chartevents_path}")
    else:
        print(f"⚠️  Chartevents file not found: {data_paths['chartevents']}")
        print("   Skipping chart events extraction")
    
    # Step 1.4: Load and extract lab events
    print("\n" + "="*80)
    print("Step 1.4: Loading and extracting lab events")
    print("="*80)
    
    if os.path.exists(data_paths['labevents']):
        print(f"\nLoading labevents from {data_paths['labevents']}...")
        file_size_gb = os.path.getsize(data_paths['labevents']) / (1024**3)
        print(f"File size: {file_size_gb:.1f} GB")
        print("⚠️  This is also a large file. Processing will take 10-20 minutes...")
        
        # Load in chunks
        chunk_size = 500000
        labevents_list = []
        chunk_num = 0
        total_rows = 0
        kept_rows = 0
        
        print("\nProcessing chunks (500K rows each)...")
        print("⚠️  Using error-tolerant mode (will skip malformed rows)")
        
        for chunk in pd.read_csv(data_paths['labevents'], 
                                 chunksize=chunk_size,
                                 on_bad_lines='warn',
                                 engine='c',
                                 low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            
            chunk_cohort = chunk[chunk['hadm_id'].isin(cohort_hadm_ids)]
            kept_rows += len(chunk_cohort)
            
            if len(chunk_cohort) > 0:
                labevents_list.append(chunk_cohort)
            
            if chunk_num % 10 == 0:
                print(f"  Processed chunk {chunk_num}: {total_rows:,} rows total, {kept_rows:,} kept ({kept_rows/total_rows*100:.1f}%)")
        
        print(f"\n✓ Finished loading: {total_rows:,} total rows, {kept_rows:,} kept")
        
        labevents = pd.concat(labevents_list, ignore_index=True)
        print(f"✓ Concatenated {len(labevents):,} lab events for cohort")
        
        # Extract events in time window
        labevents_cohort = extract_events_for_cohort(
            labevents, cohort, time_col='charttime', window_hours=window_hours
        )
        
        # Save
        labevents_path = os.path.join(output_dir, 'labevents_raw.parquet')
        labevents_cohort.to_parquet(labevents_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Saved to {labevents_path}")
    else:
        print(f"⚠️  Labevents file not found: {data_paths['labevents']}")
        print("   Skipping lab events extraction")
    
    # Step 1.5: Load and extract prescriptions
    print("\n" + "="*80)
    print("Step 1.5: Loading and extracting prescriptions")
    print("="*80)
    
    if os.path.exists(data_paths['prescriptions']):
        print(f"\nLoading prescriptions...")
        prescriptions = pd.read_csv(data_paths['prescriptions'])
        print(f"✓ Loaded {len(prescriptions):,} prescriptions")
        
        # Filter to cohort
        prescriptions_cohort = prescriptions[
            prescriptions['hadm_id'].isin(cohort['hadm_id'])
        ].copy()
        print(f"✓ Prescriptions for cohort: {len(prescriptions_cohort):,}")
        
        # Extract in time window
        prescriptions_cohort = extract_events_for_cohort(
            prescriptions_cohort, cohort, time_col='starttime', window_hours=window_hours
        )
        
        # Save
        prescriptions_path = os.path.join(output_dir, 'prescriptions_raw.parquet')
        prescriptions_cohort.to_parquet(prescriptions_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Saved to {prescriptions_path}")
    else:
        print(f"⚠️  Prescriptions file not found: {data_paths['prescriptions']}")
        print("   Skipping prescriptions extraction")
    
    # Print final summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Cohort:           {len(cohort):>10,} admissions")
    print(f"  Readmit rate:   {cohort['readmit_30d'].mean()*100:>9.2f}%")
    
    if os.path.exists(os.path.join(output_dir, 'chartevents_raw.parquet')):
        ce = pd.read_parquet(os.path.join(output_dir, 'chartevents_raw.parquet'))
        print(f"Chart events:     {len(ce):>10,} events")
    
    if os.path.exists(os.path.join(output_dir, 'labevents_raw.parquet')):
        le = pd.read_parquet(os.path.join(output_dir, 'labevents_raw.parquet'))
        print(f"Lab events:       {len(le):>10,} events")
    
    if os.path.exists(os.path.join(output_dir, 'prescriptions_raw.parquet')):
        pr = pd.read_parquet(os.path.join(output_dir, 'prescriptions_raw.parquet'))
        print(f"Prescriptions:    {len(pr):>10,} prescriptions")
    
    print(f"\n✓ Step 1 complete! Data saved to {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
