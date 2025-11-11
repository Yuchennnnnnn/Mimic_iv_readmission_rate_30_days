"""
Step 4: Compute Masks, Deltas, and Apply Imputation
"""

import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from utils import (
    load_config, compute_mask, compute_delta,
    forward_fill, median_impute, compute_global_medians
)


def process_sample(sample, global_medians, feature_names):
    """
    Process one sample: compute masks, deltas, forward-fill, median imputation
    
    Args:
        sample: Dict with 'values' array of shape (n_hours, n_features)
        global_medians: Dict mapping feature name to global median
        feature_names: List of feature names
    
    Returns:
        Updated sample with 'masks', 'deltas', and imputed 'values'
    """
    values = sample['values']  # Shape: (n_hours, n_features)
    n_hours, n_features = values.shape
    
    masks = np.zeros((n_hours, n_features))
    deltas = np.zeros((n_hours, n_features))
    values_imputed = values.copy()
    
    # Process each feature independently
    for f_idx, fname in enumerate(feature_names):
        feature_vals = values[:, f_idx]
        
        # Step 1: Compute mask
        mask = compute_mask(feature_vals)
        masks[:, f_idx] = mask
        
        # Step 2: Compute delta
        delta = compute_delta(mask)
        deltas[:, f_idx] = delta
        
        # Step 3: Forward fill
        filled = forward_fill(feature_vals, mask)
        
        # Step 4: Median imputation for remaining NaNs
        if fname in global_medians:
            filled = median_impute(filled, global_medians[fname])
        else:
            filled = median_impute(filled, 0.0)  # Default
        
        values_imputed[:, f_idx] = filled
    
    # Update sample
    sample['values'] = values_imputed
    sample['masks'] = masks
    sample['deltas'] = deltas
    
    return sample


def main():
    """Main execution"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 4: Compute Features")
    print("="*80)
    
    # Load config
    config = load_config()
    output_dir = config['paths']['output_dir']
    
    # Load binned time series from Step 3
    input_path = os.path.join(output_dir, 'timeseries_binned.pkl')
    print(f"\nLoading data from {input_path}...")
    
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data_list = data_dict['data']
    feature_names = data_dict['feature_names']
    n_hours = data_dict['n_hours']
    n_features = data_dict['n_features']
    
    print(f"✓ Loaded {len(data_list):,} samples")
    print(f"✓ Shape per sample: ({n_hours}, {n_features})")
    
    # Compute global medians from ALL data (for imputation)
    print("\nComputing global medians...")
    global_medians = compute_global_medians(data_list, feature_names)
    
    print(f"✓ Computed medians for {len(global_medians)} features")
    print("\nExample medians:")
    for i, (fname, median) in enumerate(list(global_medians.items())[:10]):
        print(f"  {fname:30s}: {median:.2f}")
    if len(global_medians) > 10:
        print(f"  ... and {len(global_medians) - 10} more")
    
    # Process all samples
    print(f"\nProcessing {len(data_list):,} samples...")
    for i in tqdm(range(len(data_list))):
        data_list[i] = process_sample(data_list[i], global_medians, feature_names)
    
    # Validate
    print("\nValidating...")
    for sample in data_list:
        assert not np.any(np.isnan(sample['values'])), "NaNs remain after imputation!"
        assert sample['masks'].shape == sample['values'].shape
        assert sample['deltas'].shape == sample['values'].shape
    print("✓ All samples validated")
    
    # Save processed data
    output_path = os.path.join(output_dir, 'timeseries_processed.pkl')
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': data_list,
            'feature_names': feature_names,
            'global_medians': global_medians,
            'n_hours': n_hours,
            'n_features': n_features
        }, f)
    
    print(f"✓ Saved {len(data_list):,} processed samples")
    
    # Print statistics
    print("\nFinal statistics:")
    all_masks = np.array([s['masks'] for s in data_list])
    obs_rate = all_masks.mean() * 100
    print(f"  Observation rate: {obs_rate:.2f}%")
    
    all_deltas = np.array([s['deltas'] for s in data_list])
    avg_delta = all_deltas.mean()
    max_delta = all_deltas.max()
    print(f"  Average delta: {avg_delta:.2f} hours")
    print(f"  Max delta: {max_delta:.2f} hours")
    
    print("\n✓ Step 4 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
