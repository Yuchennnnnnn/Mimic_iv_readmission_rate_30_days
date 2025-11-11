"""
Step 6: Save Final Output
Saves data as pickle + parquet index for easy access
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from utils import load_config, validate_sample


def create_index_dataframe(data_list):
    """
    Create Parquet index with metadata for each sample
    
    Returns:
        DataFrame with columns: hadm_id, subject_id, admittime, readmit_30d, 
                                 anchor_year_group, file_idx
    """
    records = []
    for idx, sample in enumerate(data_list):
        records.append({
            'file_idx': idx,
            'hadm_id': sample['hadm_id'],
            'subject_id': sample['subject_id'],
            'admittime': sample['admittime'],
            'readmit_30d': sample['readmit_30d'],
            'anchor_year_group': sample['anchor_year_group']
        })
    
    return pd.DataFrame(records)


def main():
    """Main execution"""
    print("="*80)
    print("MIMIC-IV Preprocessing - Step 6: Save Final Output")
    print("="*80)
    
    # Load config
    config = load_config()
    output_dir = config['paths']['output_dir']
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'-'*80}")
        print(f"Processing {split_name} split")
        print(f"{'-'*80}")
        
        # Load data
        input_path = os.path.join(output_dir, f'{split_name}_data.pkl')
        print(f"Loading from {input_path}...")
        
        with open(input_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        data_list = data_dict['data']
        print(f"✓ Loaded {len(data_list):,} samples")
        
        # Validate all samples
        print("Validating samples...")
        n_valid = 0
        for sample in data_list:
            if validate_sample(sample):
                n_valid += 1
        
        if n_valid != len(data_list):
            print(f"WARNING: Only {n_valid}/{len(data_list)} samples are valid!")
        else:
            print(f"✓ All {n_valid} samples validated")
        
        # Save pickle (already done in Step 5, but we can verify)
        pickle_path = os.path.join(output_dir, f'{split_name}_data.pkl')
        print(f"✓ Pickle saved at {pickle_path}")
        
        # Create and save Parquet index
        print("Creating Parquet index...")
        index_df = create_index_dataframe(data_list)
        
        index_path = os.path.join(output_dir, f'{split_name}_index.parquet')
        index_df.to_parquet(index_path, compression='gzip', index=False, engine='fastparquet')
        print(f"✓ Index saved to {index_path}")
        
        # Print sample of index
        print("\nIndex sample:")
        print(index_df.head())
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Total samples: {len(data_list):,}")
        print(f"  Readmissions: {index_df['readmit_30d'].sum():,} ({index_df['readmit_30d'].mean()*100:.2f}%)")
        print(f"  Unique patients: {index_df['subject_id'].nunique():,}")
        print(f"  Unique admissions: {index_df['hadm_id'].nunique():,}")
        
        # Print year distribution
        print(f"\nAnchor year group distribution:")
        year_counts = index_df['anchor_year_group'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count:>6,} samples ({count/len(index_df)*100:>5.2f}%)")
    
    # Create summary statistics across all splits
    print(f"\n{'='*80}")
    print("Overall Summary")
    print(f"{'='*80}")
    
    total_samples = 0
    total_readmissions = 0
    
    for split_name in ['train', 'val', 'test']:
        index_path = os.path.join(output_dir, f'{split_name}_index.parquet')
        index_df = pd.read_parquet(index_path, engine='fastparquet')
        
        n_samples = len(index_df)
        n_readmit = index_df['readmit_30d'].sum()
        
        total_samples += n_samples
        total_readmissions += n_readmit
        
        print(f"{split_name.capitalize():12s}: {n_samples:>8,} samples, {n_readmit:>6,} readmissions ({n_readmit/n_samples*100:>5.2f}%)")
    
    print(f"{'-'*80}")
    print(f"{'Total':12s}: {total_samples:>8,} samples, {total_readmissions:>6,} readmissions ({total_readmissions/total_samples*100:>5.2f}%)")
    
    print(f"\n✓ Step 6 complete!")
    print(f"✓ All data saved to {output_dir}")
    print("="*80)
    
    # Print usage example
    print("\n" + "="*80)
    print("Usage Example")
    print("="*80)
    print("""
# Load training data
import pickle
import pandas as pd

# Load pickle
with open('output/train_data.pkl', 'rb') as f:
    train_dict = pickle.load(f)

train_data = train_dict['data']
feature_names = train_dict['feature_names']

# Load index
train_index = pd.read_parquet('output/train_index.parquet', engine='fastparquet')

# Access one sample
sample = train_data[0]
print(f"Admission ID: {sample['hadm_id']}")
print(f"Values shape: {sample['values'].shape}")  # (48, n_features)
print(f"Masks shape: {sample['masks'].shape}")    # (48, n_features)
print(f"Deltas shape: {sample['deltas'].shape}")  # (48, n_features)
print(f"Label: {sample['readmit_30d']}")

# Use in PyTorch DataLoader
from torch.utils.data import Dataset, DataLoader
import torch

class MIMICDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'values': torch.FloatTensor(sample['values']),
            'masks': torch.FloatTensor(sample['masks']),
            'deltas': torch.FloatTensor(sample['deltas']),
            'label': torch.LongTensor([sample['readmit_30d']])
        }

dataset = MIMICDataset(train_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """)
    print("="*80)


if __name__ == "__main__":
    main()
