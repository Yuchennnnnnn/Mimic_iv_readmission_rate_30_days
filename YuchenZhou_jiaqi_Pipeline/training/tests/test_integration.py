"""
Simple test to verify the pipeline works end-to-end with synthetic data.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, set_seed


def create_synthetic_dataset(n_samples=1000, seed=42):
    """
    Create a synthetic readmission dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(seed)
    
    # Generate features
    data = {
        'subject_id': np.arange(n_samples),
        'hadm_id': np.arange(n_samples) + 10000,
        'admittime': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'dischtime': pd.date_range('2020-01-02', periods=n_samples, freq='H'),
        
        # Categorical features
        'gender': np.random.choice(['M', 'F'], n_samples),
        'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Other'], n_samples),
        'admission_type': np.random.choice(['ELECTIVE', 'URGENT', 'EMERGENCY'], n_samples),
        'discharge_location': np.random.choice(['HOME', 'SNF', 'REHAB', 'HOSPICE'], n_samples),
        'marital_status': np.random.choice(['SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED'], n_samples),
        'language': np.random.choice(['ENGLISH', 'SPANISH', 'OTHER'], n_samples),
        'last_service': np.random.choice(['MED', 'SURG', 'CARD', 'OMED', 'ORTHO'], n_samples),
        'admission_location': np.random.choice(['EMERGENCY', 'CLINIC', 'TRANSFER'], n_samples),
        
        # Numeric features
        'anchor_age': np.random.randint(18, 90, n_samples),
        'length_of_stay': np.random.exponential(5, n_samples),
        'num_diagnoses': np.random.poisson(5, n_samples),
        'num_transfers': np.random.poisson(2, n_samples),
        'unique_careunits': np.random.randint(1, 5, n_samples),
        'ed_los_hours': np.random.uniform(0, 48, n_samples),
        'days_since_prev_discharge': np.random.exponential(30, n_samples),
        
        # Lab values
        'Creatinine_min': np.random.uniform(0.5, 3.0, n_samples),
        'Glucose_min': np.random.uniform(50, 200, n_samples),
        'Hemoglobin_min': np.random.uniform(7, 16, n_samples),
        
        # Binary features
        'has_prior_admission': np.random.binomial(1, 0.4, n_samples),
        'ed_visit_flag': np.random.binomial(1, 0.3, n_samples),
        'had_icu_transfer_flag': np.random.binomial(1, 0.25, n_samples),
        'died_in_hospital': np.random.binomial(1, 0.05, n_samples),
        'is_surgical_service': np.random.binomial(1, 0.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with some correlation to features
    # Make readmission more likely for:
    # - longer LOS
    # - more diagnoses
    # - certain discharge locations
    # - prior admissions
    risk_score = (
        0.05 * df['length_of_stay'] +
        0.1 * df['num_diagnoses'] +
        0.3 * df['has_prior_admission'] +
        0.2 * (df['discharge_location'] == 'SNF').astype(int) +
        0.15 * (df['admission_type'] == 'EMERGENCY').astype(int) +
        np.random.normal(0, 1, n_samples)
    )
    
    # Convert to probability and sample
    prob = 1 / (1 + np.exp(-risk_score))
    df['readmit_label'] = (prob > np.random.uniform(0, 1, n_samples)).astype(int)
    
    # Adjust to have ~20-30% readmission rate
    readmit_rate = df['readmit_label'].mean()
    if readmit_rate < 0.2:
        # Randomly flip some negatives to positives
        n_flip = int((0.25 - readmit_rate) * n_samples)
        neg_indices = df[df['readmit_label'] == 0].index
        flip_indices = np.random.choice(neg_indices, n_flip, replace=False)
        df.loc[flip_indices, 'readmit_label'] = 1
    
    print(f"\nGenerated synthetic dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {len(df.columns) - 5}")  # Exclude ID, time, label
    print(f"  Readmission rate: {df['readmit_label'].mean():.2%}")
    
    return df


if __name__ == '__main__':
    # Create synthetic data
    df = create_synthetic_dataset(1000)
    
    # Save to temp location
    output_path = '../data/synthetic_data.csv'
    os.makedirs('../data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved synthetic data to: {output_path}")
    print("\nTo test the pipeline with synthetic data:")
    print("  1. Update config.yaml:")
    print("     data:")
    print("       input_path: 'data/synthetic_data.csv'")
    print("  2. Run training:")
    print("     python src/train.py --model logistic --config config.yaml")
