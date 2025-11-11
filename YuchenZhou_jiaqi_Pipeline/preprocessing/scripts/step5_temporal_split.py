"""
Step 5: Temporal Split by Anchor Year Group
Prevents data leakage by splitting patients temporally
"""

import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict
from utils import load_config, print_statistics


def split_by_anchor_year(data_list, train_years, val_years, test_years):
    """
    Split data by anchor_year_group
    
    Args:
        data_list: List of samples
        train_years: List of year ranges for training (e.g., [2008, 2014])
        val_years: List for validation
        test_years: List for test
    
    Returns:
        train_data, val_data, test_data
    """
    train_data = []
    val_data = []
    test_data = []
    
    # Group by subject_id to ensure no patient overlap
    subject_to_samples = defaultdict(list)
    for sample in data_list:
        subject_id = sample['subject_id']
        subject_to_samples[subject_id].append(sample)
    
    # Split subjects by anchor_year_group
    for subject_id, samples in subject_to_samples.items():
        # Get anchor year group (should be same for all samples of one patient)
        anchor_year_group = samples[0]['anchor_year_group']
        
        # Parse year range (e.g., "2008 - 2010")
        year_parts = anchor_year_group.split(' - ')
        if len(year_parts) == 2:
            start_year = int(year_parts[0])
            end_year = int(year_parts[1])
            mid_year = (start_year + end_year) / 2
        else:
            # Fallback
            mid_year = 2015
        
        # Assign to split
        if train_years[0] <= mid_year < train_years[1]:
            train_data.extend(samples)
        elif val_years[0] <= mid_year < val_years[1]:
            val_data.extend(samples)
        elif test_years[0] <= mid_year <= test_years[1]:
            test_data.extend(samples)
    
    return train_data, val_data, test_data


def verify_no_patient_overlap(train_data, val_data, test_data):
    """Verify no patient appears in multiple splits"""
    train_subjects = set(s['subject_id'] for s in train_data)
    val_subjects = set(s['subject_id'] for s in val_data)
    test_subjects = set(s['subject_id'] for s in test_data)
    
    overlap_train_val = train_subjects & val_subjects
    overlap_train_test = train_subjects & test_subjects
    overlap_val_test = val_subjects & test_subjects
    
    if len(overlap_train_val) > 0:
        print(f"WARNING: {len(overlap_train_val)} patients in both train and val!")
        return False
    if len(overlap_train_test) > 0:
        print(f"WARNING: {len(overlap_train_test)} patients in both train and test!")
        return False
    if len(overlap_val_test) > 0:
        print(f"WARNING: {len(overlap_val_test)} patients in both val and test!")
        return False
    
    print("✓ No patient overlap between splits")
    return True


def main():
    """Main execution"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 5: Temporal Split")
    print("="*80)
    
    # Load config
    config = load_config()
    output_dir = config['paths']['output_dir']
    train_years = config['temporal_split']['train_years']
    val_years = config['temporal_split']['val_years']
    test_years = config['temporal_split']['test_years']
    
    print(f"\nTemporal split configuration:")
    print(f"  Train: {train_years[0]} - {train_years[1]}")
    print(f"  Val:   {val_years[0]} - {val_years[1]}")
    print(f"  Test:  {test_years[0]} - {test_years[1]}")
    
    # Load processed data from Step 4
    input_path = os.path.join(output_dir, 'timeseries_processed.pkl')
    print(f"\nLoading data from {input_path}...")
    
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data_list = data_dict['data']
    feature_names = data_dict['feature_names']
    global_medians = data_dict['global_medians']
    n_hours = data_dict['n_hours']
    n_features = data_dict['n_features']
    
    print(f"✓ Loaded {len(data_list):,} samples")
    
    # Perform temporal split
    print("\nPerforming temporal split...")
    train_data, val_data, test_data = split_by_anchor_year(
        data_list, train_years, val_years, test_years
    )
    
    print(f"✓ Train: {len(train_data):,} samples")
    print(f"✓ Val:   {len(val_data):,} samples")
    print(f"✓ Test:  {len(test_data):,} samples")
    
    # Verify no overlap
    print("\nVerifying no patient overlap...")
    verify_no_patient_overlap(train_data, val_data, test_data)
    
    # Print statistics
    print_statistics(train_data, "Train")
    print_statistics(val_data, "Validation")
    print_statistics(test_data, "Test")
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}_data.pkl')
        print(f"Saving {split_name} to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'data': split_data,
                'feature_names': feature_names,
                'global_medians': global_medians,
                'n_hours': n_hours,
                'n_features': n_features
            }, f)
        
        print(f"✓ Saved {len(split_data):,} samples")
    
    print("\n✓ Step 5 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
