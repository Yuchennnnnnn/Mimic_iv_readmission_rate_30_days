"""
Step 3: Create Fixed-Length Time Series (48-hour bins)
Converts irregular event data to fixed hourly bins
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import pickle
from utils import (
    load_config, bin_to_hour, aggregate_to_bins,
    ITEMID_TO_FEATURE, apply_outlier_bounds
)


def load_data(output_dir):
    """Load preprocessed data from Step 2"""
    cohort = pd.read_parquet(os.path.join(output_dir, 'cohort.parquet'), engine='fastparquet')
    chartevents = pd.read_parquet(os.path.join(output_dir, 'chartevents_clean.parquet'), engine='fastparquet')
    labevents = pd.read_parquet(os.path.join(output_dir, 'labevents_clean.parquet'), engine='fastparquet')
    
    # Convert time columns to datetime
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    labevents['charttime'] = pd.to_datetime(labevents['charttime'])
    
    # Prescriptions is optional
    prescriptions_path = os.path.join(output_dir, 'prescriptions_clean.parquet')
    if os.path.exists(prescriptions_path):
        prescriptions = pd.read_parquet(prescriptions_path, engine='fastparquet')
        if 'starttime' in prescriptions.columns:
            prescriptions['starttime'] = pd.to_datetime(prescriptions['starttime'])
        if 'stoptime' in prescriptions.columns:
            prescriptions['stoptime'] = pd.to_datetime(prescriptions['stoptime'])
    else:
        prescriptions = pd.DataFrame()
    
    return cohort, chartevents, labevents, prescriptions


def create_timeseries_for_admission(
    hadm_id,
    admittime,
    chartevents_adm,
    labevents_adm,
    prescriptions_adm,
    n_hours=48
):
    """
    Create fixed-length time series for one admission
    
    Returns:
        Dictionary with feature_name -> array of shape (n_hours,)
    """
    timeseries = {}
    
    # Process chart events
    for feature_name in chartevents_adm['feature_name'].unique():
        feature_data = chartevents_adm[chartevents_adm['feature_name'] == feature_name]
        
        # Apply outlier bounds
        values = []
        timestamps = []
        for _, row in feature_data.iterrows():
            val = apply_outlier_bounds(row['valuenum'], feature_name)
            if val is not None:
                values.append(val)
                timestamps.append(row['charttime'])
        
        if len(values) > 0:
            # Aggregate to hourly bins
            binned = aggregate_to_bins(values, timestamps, admittime, n_hours, agg_func='median')
            timeseries[feature_name] = binned
    
    # Process lab events
    for feature_name in labevents_adm['feature_name'].unique():
        feature_data = labevents_adm[labevents_adm['feature_name'] == feature_name]
        
        values = []
        timestamps = []
        for _, row in feature_data.iterrows():
            val = apply_outlier_bounds(row['valuenum'], feature_name)
            if val is not None:
                values.append(val)
                timestamps.append(row['charttime'])
        
        if len(values) > 0:
            binned = aggregate_to_bins(values, timestamps, admittime, n_hours, agg_func='median')
            timeseries[feature_name] = binned
    
    # Process prescriptions (convert to binary indicators per hour)
    # Only process if prescriptions data exists
    if not prescriptions_adm.empty and 'medication_category' in prescriptions_adm.columns:
        for med_category in prescriptions_adm['medication_category'].unique():
            feature_name = f'med_{med_category}'
            med_data = prescriptions_adm[prescriptions_adm['medication_category'] == med_category]
            
            # Create binary indicator array
            binary = np.zeros(n_hours)
            for _, row in med_data.iterrows():
                start_hour = bin_to_hour(row['starttime'], admittime, n_hours)
                stop_hour = bin_to_hour(row['stoptime'], admittime, n_hours) if pd.notna(row['stoptime']) else n_hours
                
                if start_hour is not None:
                    stop_hour = min(stop_hour if stop_hour is not None else n_hours, n_hours)
                    binary[start_hour:stop_hour] = 1
            
            timeseries[feature_name] = binary
    
    return timeseries


def create_feature_matrix(timeseries_dict, feature_names, n_hours=48):
    """
    Convert dictionary of time series to matrix
    
    Args:
        timeseries_dict: Dict of feature_name -> array
        feature_names: Ordered list of all feature names
        n_hours: Number of time steps
    
    Returns:
        Matrix of shape (n_hours, n_features)
    """
    n_features = len(feature_names)
    matrix = np.full((n_hours, n_features), np.nan)
    
    for f_idx, fname in enumerate(feature_names):
        if fname in timeseries_dict:
            matrix[:, f_idx] = timeseries_dict[fname]
    
    return matrix


def main():
    """Main execution"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 3: Create Time Series")
    print("="*80)
    
    # Load config
    config = load_config()
    output_dir = config['paths']['output_dir']
    n_hours = config['preprocessing']['time_window_hours']
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Time window: {n_hours} hours")
    
    # Load data
    print("\nLoading data...")
    cohort, chartevents, labevents, prescriptions = load_data(output_dir)
    print(f"✓ Loaded cohort: {len(cohort):,} admissions")
    print(f"✓ Loaded chartevents: {len(chartevents):,} events")
    print(f"✓ Loaded labevents: {len(labevents):,} events")
    print(f"✓ Loaded prescriptions: {len(prescriptions):,} prescriptions")
    
    # Get all unique feature names
    chart_features = sorted(chartevents['feature_name'].unique())
    lab_features = sorted(labevents['feature_name'].unique())
    
    # Only get medication features if prescriptions data exists
    if not prescriptions.empty and 'medication_category' in prescriptions.columns:
        med_features = sorted([f'med_{cat}' for cat in prescriptions['medication_category'].unique()])
    else:
        med_features = []
    
    feature_names = chart_features + lab_features + med_features
    n_features = len(feature_names)
    
    print(f"\nTotal features: {n_features}")
    print(f"  Chart features: {len(chart_features)}")
    print(f"  Lab features: {len(lab_features)}")
    print(f"  Medication features: {len(med_features)}")
    
    # Process each admission
    print(f"\nProcessing {len(cohort):,} admissions...")
    
    data_list = []
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort)):
        hadm_id = row['hadm_id']
        admittime = row['admittime']
        readmit_30d = row['readmit_30d']
        
        # Get events for this admission
        chartevents_adm = chartevents[chartevents['hadm_id'] == hadm_id]
        labevents_adm = labevents[labevents['hadm_id'] == hadm_id]
        
        # Only filter prescriptions if data exists
        if not prescriptions.empty:
            prescriptions_adm = prescriptions[prescriptions['hadm_id'] == hadm_id]
        else:
            prescriptions_adm = pd.DataFrame()
        
        # Create time series
        timeseries_dict = create_timeseries_for_admission(
            hadm_id, admittime,
            chartevents_adm, labevents_adm, prescriptions_adm,
            n_hours
        )
        
        # Convert to matrix
        matrix = create_feature_matrix(timeseries_dict, feature_names, n_hours)
        
        # Store
        sample = {
            'hadm_id': hadm_id,
            'subject_id': row['subject_id'],
            'admittime': admittime,
            'values': matrix,  # Shape: (n_hours, n_features)
            'readmit_30d': readmit_30d,
            'anchor_year_group': row['anchor_year_group']
        }
        
        data_list.append(sample)
    
    # Save
    output_path = os.path.join(output_dir, 'timeseries_binned.pkl')
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': data_list,
            'feature_names': feature_names,
            'n_hours': n_hours,
            'n_features': n_features
        }, f)
    
    # Also save feature names separately
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for fname in feature_names:
            f.write(f"{fname}\n")
    
    print(f"✓ Saved {len(data_list):,} samples")
    print(f"✓ Feature names saved to {feature_names_path}")
    
    # Print statistics
    print("\nData statistics:")
    print(f"  Shape per sample: ({n_hours}, {n_features})")
    
    all_matrices = np.array([s['values'] for s in data_list])
    obs_rate = (~np.isnan(all_matrices)).mean() * 100
    print(f"  Overall observation rate: {obs_rate:.2f}%")
    
    print("\n✓ Step 3 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
