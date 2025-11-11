# MIMIC-IV Temporal Preprocessing Pipeline

## Overview

This preprocessing pipeline creates fixed-length time-series data from MIMIC-IV for 30-day readmission prediction. It handles:

1. **Cohort Definition**: Load MIMIC-IV CSVs and identify admissions with 30-day readmission labels
2. **Event Extraction**: Filter chartevents, labevents, and prescriptions for cohort
3. **Data Cleaning**: Unit standardization and quality checks
4. **Temporal Binning**: First 48 hours after admission, 1-hour bins (48 timesteps)
5. **Feature Engineering**: Masks, deltas (time-since-last-observed), forward-fill imputation
6. **Outlier Handling**: Clinically-informed bounds + median imputation
7. **Temporal Split**: Patient-level split by anchor_year_group (prevents leakage)
8. **Output Format**: Pickled list of dicts + Parquet index

---

## Folder Structure

```
preprocessing/
├── README.md                          # This file
├── QUICKSTART.md                      # 3-step quick start guide
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration parameters
├── run_all.sh                         # Execute full pipeline
│
├── scripts/                           # Python processing scripts
│   ├── step1_load_data.py            # Load CSVs and create cohort
│   ├── step2_clean_units.py          # Clean and standardize units
│   ├── step3_create_timeseries.py    # Bin into 48 1-hour windows
│   ├── step4_compute_features.py     # Compute masks, deltas, imputation
│   ├── step5_temporal_split.py       # Split by anchor_year_group
│   ├── step6_save_output.py          # Save as pickle + parquet
│   └── utils.py                      # Shared utility functions
│
└── output/                            # Generated data (gitignored)
    ├── cohort.parquet
    ├── chartevents_raw.parquet
    ├── labevents_raw.parquet
    ├── prescriptions_raw.parquet
    ├── timeseries_binned.pkl
    ├── train_data.pkl
    ├── val_data.pkl
    ├── test_data.pkl
    ├── train_index.parquet
    ├── val_index.parquet
    └── test_index.parquet
```

---

## Quick Start

### 1. Setup BigQuery Access

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Install Dependencies

```bash
cd preprocessing
pip install -r requirements.txt
```

### 3. Configure Parameters

Edit `config.yaml`:
- Set your BigQuery project ID
- Adjust clinical outlier bounds
- Set temporal split ratios

### 4. Run Full Pipeline

```bash
# Option 1: Run all steps sequentially
bash run_all.sh

# Option 2: Run steps individually
python scripts/step1_run_bigquery.py
python scripts/step2_clean_units.py
python scripts/step3_create_timeseries.py
python scripts/step4_compute_features.py
python scripts/step5_temporal_split.py
python scripts/step6_save_output.py
```

---

## Pipeline Details

### Step 1: Cohort Definition (BigQuery SQL)

**Query**: `sql/01_cohort_definition.sql`

Creates cohort with:
- `subject_id`, `hadm_id`, `admittime`, `dischtime`
- `readmit_30d` label (1 if readmitted within 30 days, 0 otherwise)
- Exclusions: age < 18, died in hospital, missing discharge time
- Output: ~200K admissions

**Key Logic:**
```sql
WITH next_admit AS (
  SELECT 
    hadm_id,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime,
    dischtime
  FROM admissions
)
SELECT 
  ...,
  CASE 
    WHEN DATE_DIFF(next_admittime, dischtime, DAY) <= 30 THEN 1 
    ELSE 0 
  END AS readmit_30d
```

### Step 2: Extract Events (BigQuery SQL)

**Queries**: `sql/02_extract_chartevents.sql`, `03_extract_labevents.sql`, `04_extract_prescriptions.sql`

Extracts only events for cohort `hadm_id`s to reduce data size:
- **Chart Events**: Vitals (HR, BP, SpO2, Temp), I/O, GCS
- **Lab Events**: CBC, metabolic panel, coagulation
- **Prescriptions**: Start/stop times, dosage

Filters:
- Only first 48 hours after `admittime`
- Valid itemid (remove rare items)
- Non-null values

### Step 3: Clean and Standardize Units

**Script**: `step2_clean_units.py`

- **Unit conversion**: mmHg, bpm, °C/°F → standardized
- **Duplicate removal**: Same itemid at same timestamp
- **Value validation**: Remove physiologically impossible values
- **Item mapping**: Map itemid → meaningful variable names

Example mappings:
```python
VITALS_MAP = {
    220045: 'heart_rate',      # bpm
    220050: 'sbp',             # mmHg
    220051: 'dbp',             # mmHg
    220179: 'spo2',            # %
    223761: 'temperature',     # °C
}
```

### Step 4: Create Fixed-Length Time Series

**Script**: `step3_create_timeseries.py`

- **Time window**: 0-48 hours after admission
- **Binning**: 1-hour bins → 48 timesteps
- **Aggregation**: Median value within each bin
- **Shape**: (num_admissions, 48_timesteps, num_features)

Example for one admission:
```python
{
    'hadm_id': 12345,
    'readmit_30d': 1,
    'timeseries': np.array([48, 120]),  # 48 hours × 120 features
    'admittime': '2019-01-15 10:30:00'
}
```

### Step 5: Compute Masks and Deltas

**Script**: `step4_compute_features.py`

For each feature at each timestep:

1. **Mask** (observation indicator):
   ```python
   mask[t, f] = 1 if feature f observed at time t, else 0
   ```

2. **Delta** (time since last observation):
   ```python
   delta[t, f] = hours since last observed value of feature f
   ```

3. **Forward-fill imputation**:
   ```python
   if mask[t, f] == 0:
       value[t, f] = value[t-1, f]  # carry forward
       delta[t, f] = delta[t-1, f] + 1
   ```

4. **Median imputation** (if never observed):
   ```python
   if all(mask[:, f] == 0):
       value[:, f] = global_median[f]
       delta[:, f] = 48  # max delta
   ```

Final output per admission:
```python
{
    'hadm_id': 12345,
    'values': np.array([48, 120]),   # Imputed values
    'masks': np.array([48, 120]),    # 0/1 observation indicators
    'deltas': np.array([48, 120]),   # Hours since last seen
    'readmit_30d': 1
}
```

### Step 6: Outlier Removal

**Script**: `step4_compute_features.py` (integrated)

Clinically-informed bounds:
```python
CLINICAL_BOUNDS = {
    'heart_rate': (20, 250),
    'sbp': (40, 280),
    'dbp': (20, 200),
    'spo2': (50, 100),
    'temperature': (28, 44),  # °C
    'glucose': (20, 800),
    'creatinine': (0.1, 25),
    'hemoglobin': (2, 25),
    'wbc': (0.1, 100),
}
```

Values outside bounds → treated as missing → median imputation

### Step 7: Temporal Split

**Script**: `step5_temporal_split.py`

**Why temporal split?** Prevents data leakage by ensuring:
- Training patients are from earlier years
- Validation/test patients are from later years
- Models tested on "future" patients

**Method**: Split by `anchor_year_group` (MIMIC-IV's anonymized year)

```python
# Example split
train: anchor_year_group in [2008-2011, 2011-2014]  → 60%
val:   anchor_year_group in [2014-2017]              → 20%
test:  anchor_year_group in [2017-2019]              → 20%
```

**Key:** Each patient's ALL admissions go to same split (no patient overlap)

### Step 8: Save Output

**Script**: `step6_save_output.py`

Two output formats:

1. **Pickled list of dicts** (for fast loading in training):
   ```python
   # train_data.pkl
   [
       {'hadm_id': 123, 'values': array([48,120]), 'masks': ..., 'deltas': ..., 'label': 1},
       {'hadm_id': 456, 'values': array([48,120]), 'masks': ..., 'deltas': ..., 'label': 0},
       ...
   ]
   ```

2. **Parquet index** (for metadata, joining, analysis):
   ```python
   # train_index.parquet
   | hadm_id | subject_id | admittime | readmit_30d | anchor_year_group | file_idx |
   ```

**Why this format?**
- Pickle: Fast random access during training
- Parquet: SQL-like queries, joins with other MIMIC tables
- `file_idx`: Maps back to position in pickle file

---

## Data Statistics (Expected)

After processing MIMIC-IV:

```
Total admissions: ~205,000
After exclusions: ~195,000

Readmission rate: 26.7%

Temporal split:
├── Train:      117,000 admissions (60%)
├── Validation:  39,000 admissions (20%)
└── Test:        39,000 admissions (20%)

Features:
├── Chart events: ~80 features (vitals, I/O, scores)
├── Lab events:   ~35 features (CBC, chemistry)
└── Medications:   ~5 features (counts per hour)
Total:            ~120 features

Time series shape: (num_admissions, 48, 120)
```

---

## Configuration Options

Edit `config.yaml`:

```yaml
bigquery:
  project_id: "your-project-id"
  dataset: "physionet-data.mimiciv_hosp"
  
preprocessing:
  time_window_hours: 48
  bin_size_hours: 1
  min_age: 18
  readmit_window_days: 30
  
clinical_bounds:
  heart_rate: [20, 250]
  sbp: [40, 280]
  # ... etc
  
temporal_split:
  train_years: [2008, 2014]
  val_years: [2014, 2017]
  test_years: [2017, 2019]
  
output:
  save_pickle: true
  save_parquet: true
  compression: "gzip"
```

---

## Usage Example

```python
# Load preprocessed data
import pickle
import pandas as pd

# Load training data
with open('output/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Load index for metadata
train_index = pd.read_parquet('output/train_index.parquet')

# Access one sample
sample = train_data[0]
print(f"Admission ID: {sample['hadm_id']}")
print(f"Time series shape: {sample['values'].shape}")  # (48, 120)
print(f"Mask shape: {sample['masks'].shape}")          # (48, 120)
print(f"Delta shape: {sample['deltas'].shape}")        # (48, 120)
print(f"Label: {sample['readmit_30d']}")

# Create PyTorch DataLoader
from torch.utils.data import Dataset, DataLoader

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

train_dataset = MIMICDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch in train_loader:
    values = batch['values']  # (32, 48, 120)
    masks = batch['masks']    # (32, 48, 120)
    deltas = batch['deltas']  # (32, 48, 120)
    labels = batch['label']   # (32, 1)
    # ... model forward pass
```

---

## Model Integration

This preprocessing format is designed for temporal models:

### GRU-D (Gated Recurrent Unit with Decay)

```python
import torch.nn as nn

class GRUD(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.decay = nn.Linear(input_dim, input_dim)
        
    def forward(self, values, masks, deltas):
        # values: (batch, time, features)
        # masks: (batch, time, features)
        # deltas: (batch, time, features)
        
        batch_size, seq_len, feat_dim = values.shape
        h = torch.zeros(batch_size, self.hidden_dim).to(values.device)
        
        for t in range(seq_len):
            # Compute decay
            gamma = torch.exp(-torch.relu(self.decay(deltas[:, t, :])))
            
            # Apply decay to missing values
            x_t = masks[:, t, :] * values[:, t, :] + \
                  (1 - masks[:, t, :]) * gamma * values[:, t, :]
            
            # GRU step
            h = self.gru_cell(x_t, h)
        
        return h  # Final hidden state
```

### Transformer with Time-Aware Attention

```python
class TimeAwareTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_embedding = nn.Linear(1, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        
    def forward(self, values, masks, deltas):
        # Embed values
        x = self.embedding(values)  # (batch, seq, d_model)
        
        # Embed time deltas
        t = self.time_embedding(deltas.mean(dim=-1, keepdim=True))  # (batch, seq, d_model)
        
        # Combine
        x = x + t
        
        # Create attention mask for padding (use masks)
        padding_mask = (masks.sum(dim=-1) == 0)  # (batch, seq)
        
        # Transformer
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        output = self.transformer(x, src_key_padding_mask=padding_mask)
        
        return output[-1]  # Last timestep
```

---

## Quality Checks

The pipeline includes automatic quality checks:

```python
# In step6_save_output.py
def validate_output(data_list):
    """Validate preprocessed data"""
    for sample in data_list:
        # Check shapes
        assert sample['values'].shape == (48, 120)
        assert sample['masks'].shape == (48, 120)
        assert sample['deltas'].shape == (48, 120)
        
        # Check ranges
        assert np.all(sample['masks'] >= 0) and np.all(sample['masks'] <= 1)
        assert np.all(sample['deltas'] >= 0) and np.all(sample['deltas'] <= 48)
        
        # Check no NaNs after imputation
        assert not np.any(np.isnan(sample['values']))
        
        # Check label
        assert sample['readmit_30d'] in [0, 1]
    
    print("✓ All quality checks passed")
```

---

## Troubleshooting

### Issue: BigQuery Authentication Failed
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Issue: Out of Memory during Processing
Edit `config.yaml`:
```yaml
processing:
  chunk_size: 10000  # Process in smaller chunks
```

### Issue: Missing Values After Imputation
Check median computation in `step4_compute_features.py`:
```python
# Ensure global median is computed on train set only
train_medians = compute_medians(train_data)
# Apply to train, val, test
```

### Issue: Temporal Leakage Detected
Verify temporal split in `step5_temporal_split.py`:
```python
# Ensure no patient overlap
train_subjects = set(train_index.subject_id)
test_subjects = set(test_index.subject_id)
assert len(train_subjects & test_subjects) == 0
```

---

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@misc{mimic_readmission_preprocessing_2025,
  author = {Zhou, Yuchen and Wu, Jiaqi},
  title = {MIMIC-IV Temporal Preprocessing for 30-Day Readmission Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days}
}
```

---

## Contact

For questions or issues:
- GitHub Issues: [Repository Issues](https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days/issues)
- Email: yz946@duke.edu, jw933@duke.edu

---

**Last Updated**: November 10, 2025  
**Pipeline Version**: 1.0.0
