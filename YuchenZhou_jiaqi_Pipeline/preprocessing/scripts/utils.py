"""
Utility functions for MIMIC-IV temporal preprocessing pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Clinical Reference Ranges and Outlier Bounds
# ============================================================================

CLINICAL_BOUNDS = {
    # Vitals
    'heart_rate': (20, 250),
    'sbp': (40, 280),
    'dbp': (20, 200),
    'mbp': (30, 220),
    'respiratory_rate': (4, 60),
    'spo2': (50, 100),
    'temperature_c': (28, 44),
    'temperature_f': (82, 111),
    
    # Labs - Hematology
    'hemoglobin': (2, 25),
    'hematocrit': (10, 75),
    'platelets': (10, 1000),
    'wbc': (0.1, 100),
    'rbc': (1, 10),
    
    # Labs - Chemistry
    'sodium': (100, 180),
    'potassium': (1.5, 10),
    'chloride': (50, 150),
    'bicarbonate': (5, 50),
    'bun': (1, 200),
    'creatinine': (0.1, 25),
    'glucose': (20, 800),
    'calcium': (4, 20),
    'magnesium': (0.5, 10),
    'phosphate': (0.5, 15),
    
    # Labs - Liver
    'alt': (0, 5000),
    'ast': (0, 5000),
    'bilirubin_total': (0, 50),
    'alkaline_phosphatase': (0, 2000),
    'albumin': (0.5, 10),
    
    # Labs - Coagulation
    'inr': (0.5, 15),
    'pt': (5, 150),
    'ptt': (10, 200),
    
    # Labs - Blood Gas
    'po2': (20, 600),
    'pco2': (10, 150),
    'ph': (6.5, 8.0),
    
    # Labs - Cardiac
    'troponin': (0, 100),
    'ck': (0, 10000),
    'ck_mb': (0, 1000),
    
    # I/O
    'urine_output': (0, 2000),  # mL per hour
}

# Itemid to feature name mapping (MIMIC-IV)
ITEMID_TO_FEATURE = {
    # Vitals
    220045: 'heart_rate',
    220050: 'sbp',
    220051: 'dbp',
    220052: 'mbp',
    220179: 'sbp_ni',  # Non-invasive
    220180: 'dbp_ni',
    220181: 'mbp_ni',
    220210: 'respiratory_rate',
    223835: 'fio2',
    220277: 'spo2',
    223761: 'temperature_f',
    223762: 'temperature_c',
    220615: 'respiratory_rate_total',
    224690: 'respiratory_rate_spont',
    
    # GCS
    220739: 'gcs_eye',
    223900: 'gcs_verbal',
    223901: 'gcs_motor',
    
    # Glucose (chart)
    220621: 'glucose_serum',
    225664: 'glucose_finger',
    
    # Urine Output
    226627: 'urine_output',
    226631: 'urine_output',
    
    # Labs - CBC
    51279: 'rbc',
    51222: 'hemoglobin',
    51248: 'hematocrit',
    51265: 'platelets',
    51301: 'wbc',
    
    # Labs - Chemistry
    50912: 'creatinine',
    50971: 'potassium',
    50983: 'sodium',
    50902: 'chloride',
    50882: 'bicarbonate',
    50813: 'lactate',
    50931: 'glucose',
    51006: 'bun',
    50893: 'calcium',
    50960: 'magnesium',
    50970: 'phosphate',
    
    # Labs - Liver
    50861: 'alt',
    50878: 'ast',
    50885: 'bilirubin_total',
    50863: 'alkaline_phosphatase',
    50862: 'albumin',
    
    # Labs - Coagulation
    51237: 'inr',
    51274: 'pt',
    51275: 'ptt',
    
    # Labs - Blood Gas
    50821: 'po2',
    50818: 'pco2',
    50820: 'ph',
    
    # Labs - Cardiac
    51003: 'troponin',
    50911: 'ck',
    50910: 'ck_mb',
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_path: str = '../config.yaml') -> Dict:
    """Load configuration from YAML file"""
    import yaml
    import os
    
    # If running from scripts directory, go up one level
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def convert_temperature(value: float, unit: str) -> float:
    """Convert temperature to Celsius"""
    if unit.upper() in ['F', 'FAHRENHEIT']:
        return (value - 32) * 5/9
    return value


def standardize_units(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Standardize units for a given feature"""
    df = df.copy()
    
    # Temperature conversion
    if 'temperature' in feature_name:
        mask_f = df['valueuom'].str.upper().isin(['F', 'FAHRENHEIT'])
        df.loc[mask_f, 'valuenum'] = (df.loc[mask_f, 'valuenum'] - 32) * 5/9
        df.loc[mask_f, 'valueuom'] = 'C'
    
    return df


def apply_outlier_bounds(value: float, feature_name: str) -> Optional[float]:
    """
    Apply clinical bounds to detect outliers
    Returns None if value is outside bounds, otherwise returns value
    """
    if feature_name not in CLINICAL_BOUNDS:
        return value
    
    lower, upper = CLINICAL_BOUNDS[feature_name]
    if lower <= value <= upper:
        return value
    return None


def compute_gcs_total(gcs_eye: float, gcs_verbal: float, gcs_motor: float) -> float:
    """Compute total GCS score from components"""
    total = 0
    if not np.isnan(gcs_eye):
        total += gcs_eye
    if not np.isnan(gcs_verbal):
        total += gcs_verbal
    if not np.isnan(gcs_motor):
        total += gcs_motor
    return total if total > 0 else np.nan


def bin_to_hour(timestamp: pd.Timestamp, admittime: pd.Timestamp, n_hours: int = 48) -> Optional[int]:
    """
    Bin a timestamp to hour index (0 to n_hours-1)
    Returns None if outside the time window
    """
    hours_since_admit = (timestamp - admittime).total_seconds() / 3600
    hour_bin = int(np.floor(hours_since_admit))
    
    if 0 <= hour_bin < n_hours:
        return hour_bin
    return None


def aggregate_to_bins(
    values: List[float],
    timestamps: List[pd.Timestamp],
    admittime: pd.Timestamp,
    n_hours: int = 48,
    agg_func: str = 'median'
) -> np.ndarray:
    """
    Aggregate irregular time series to fixed hourly bins
    
    Args:
        values: List of values
        timestamps: List of timestamps
        admittime: Admission time
        n_hours: Number of hourly bins
        agg_func: Aggregation function ('median', 'mean', 'min', 'max', 'last')
    
    Returns:
        Array of shape (n_hours,) with aggregated values (NaN for missing bins)
    """
    binned = np.full(n_hours, np.nan)
    
    # Group values by hour bin
    hour_bins = {}
    for val, ts in zip(values, timestamps):
        hour_bin = bin_to_hour(ts, admittime, n_hours)
        if hour_bin is not None:
            if hour_bin not in hour_bins:
                hour_bins[hour_bin] = []
            hour_bins[hour_bin].append(val)
    
    # Aggregate within each bin
    for hour_bin, vals in hour_bins.items():
        if agg_func == 'median':
            binned[hour_bin] = np.median(vals)
        elif agg_func == 'mean':
            binned[hour_bin] = np.mean(vals)
        elif agg_func == 'min':
            binned[hour_bin] = np.min(vals)
        elif agg_func == 'max':
            binned[hour_bin] = np.max(vals)
        elif agg_func == 'last':
            binned[hour_bin] = vals[-1]  # Most recent value
        else:
            binned[hour_bin] = np.median(vals)  # Default
    
    return binned


def compute_mask(values: np.ndarray) -> np.ndarray:
    """
    Compute observation mask
    Returns 1 where observed, 0 where missing
    """
    return (~np.isnan(values)).astype(float)


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """
    Compute time since last observation (in hours)
    
    Args:
        mask: Binary mask (1 = observed, 0 = missing)
    
    Returns:
        Array of same shape with hours since last observation
    """
    n_timesteps = len(mask)
    delta = np.zeros(n_timesteps)
    
    last_observed = -1
    for t in range(n_timesteps):
        if mask[t] == 1:
            last_observed = t
            delta[t] = 0
        else:
            if last_observed >= 0:
                delta[t] = t - last_observed
            else:
                delta[t] = t  # No previous observation
    
    return delta


def forward_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Forward-fill missing values
    """
    values = values.copy()
    last_value = np.nan
    
    for t in range(len(values)):
        if mask[t] == 1:
            last_value = values[t]
        else:
            if not np.isnan(last_value):
                values[t] = last_value
    
    return values


def median_impute(values: np.ndarray, global_median: float) -> np.ndarray:
    """
    Impute remaining NaNs with global median
    """
    values = values.copy()
    values[np.isnan(values)] = global_median
    return values


def compute_global_medians(data_list: List[Dict], feature_names: List[str]) -> Dict[str, float]:
    """
    Compute global median for each feature across all samples
    Used for median imputation
    
    Args:
        data_list: List of samples, each with 'values' array of shape (n_hours, n_features)
        feature_names: List of feature names
    
    Returns:
        Dictionary mapping feature name to global median
    """
    all_values = {fname: [] for fname in feature_names}
    
    for sample in data_list:
        values = sample['values']  # Shape: (n_hours, n_features)
        for f_idx, fname in enumerate(feature_names):
            feature_vals = values[:, f_idx]
            observed = feature_vals[~np.isnan(feature_vals)]
            all_values[fname].extend(observed.tolist())
    
    medians = {}
    for fname in feature_names:
        if len(all_values[fname]) > 0:
            medians[fname] = np.median(all_values[fname])
        else:
            medians[fname] = 0.0  # Default if never observed
    
    return medians


def validate_sample(sample: Dict) -> bool:
    """
    Validate a preprocessed sample
    Returns True if valid, False otherwise
    """
    required_keys = ['hadm_id', 'values', 'masks', 'deltas', 'readmit_30d']
    
    # Check required keys
    for key in required_keys:
        if key not in sample:
            return False
    
    # Check shapes
    n_hours, n_features = sample['values'].shape
    if sample['masks'].shape != (n_hours, n_features):
        return False
    if sample['deltas'].shape != (n_hours, n_features):
        return False
    
    # Check value ranges
    if not np.all(sample['masks'] >= 0) or not np.all(sample['masks'] <= 1):
        return False
    if not np.all(sample['deltas'] >= 0):
        return False
    
    # Check no NaNs after imputation
    if np.any(np.isnan(sample['values'])):
        return False
    
    # Check label
    if sample['readmit_30d'] not in [0, 1]:
        return False
    
    return True


def print_statistics(data_list: List[Dict], split_name: str):
    """Print statistics for a data split"""
    n_samples = len(data_list)
    n_positive = sum(1 for s in data_list if s['readmit_30d'] == 1)
    pos_rate = n_positive / n_samples * 100 if n_samples > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{split_name} Split Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples:,}")
    print(f"Readmissions (positive): {n_positive:,} ({pos_rate:.2f}%)")
    print(f"No readmissions (negative): {n_samples - n_positive:,} ({100-pos_rate:.2f}%)")
    
    if n_samples > 0:
        sample = data_list[0]
        n_hours, n_features = sample['values'].shape
        print(f"Time series shape: ({n_hours}, {n_features})")
        
        # Compute missingness
        all_masks = np.array([s['masks'] for s in data_list])
        overall_observed = all_masks.mean() * 100
        print(f"Overall observation rate: {overall_observed:.2f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test functions
    print("Testing utility functions...")
    
    # Test temperature conversion
    assert abs(convert_temperature(98.6, 'F') - 37.0) < 0.1
    print("✓ Temperature conversion")
    
    # Test outlier bounds
    assert apply_outlier_bounds(80, 'heart_rate') == 80
    assert apply_outlier_bounds(300, 'heart_rate') is None
    print("✓ Outlier bounds")
    
    # Test mask and delta computation
    values = np.array([1.0, np.nan, np.nan, 2.0, np.nan])
    mask = compute_mask(values)
    assert np.array_equal(mask, [1, 0, 0, 1, 0])
    
    delta = compute_delta(mask)
    assert np.array_equal(delta, [0, 1, 2, 0, 1])
    print("✓ Mask and delta computation")
    
    # Test forward fill
    filled = forward_fill(values, mask)
    assert np.array_equal(filled, [1.0, 1.0, 1.0, 2.0, 2.0])
    print("✓ Forward fill")
    
    print("\nAll tests passed! ✓")
