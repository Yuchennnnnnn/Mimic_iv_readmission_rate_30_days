# MIMIC-IV Temporal Preprocessing Pipeline - Quick Start

## 🎯 What You Have

A complete, production-ready preprocessing pipeline that:

✅ **Cohort SQL on BigQuery** - 30-day readmission labels  
✅ **Event Extraction** - Chartevents, labevents, prescriptions for cohort  
✅ **Unit Cleaning** - Standardization and validation  
✅ **Fixed-Length Time Series** - First 48 hours, 1-hour bins  
✅ **Masks & Deltas** - Time-since-last-observed computation  
✅ **Clinical Outlier Removal** - Physiologically-informed bounds  
✅ **Median + Forward-Fill Imputation** - No NaNs in final data  
✅ **Temporal Split** - Patient-level by anchor_year_group (prevents leakage)  
✅ **Pickle + Parquet Output** - Fast loading + SQL-like queries  

## 🚀 Quick Start (3 Steps)

### 1. Setup BigQuery

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Install Dependencies

```bash
cd preprocessing
pip install -r requirements.txt
```

### 3. Configure & Run

```bash
# Edit config.yaml - Set your project ID
nano config.yaml  # Change "your-gcp-project-id" to your actual project

# Run full pipeline
./run_all.sh
```

That's it! After ~30-60 minutes, you'll have:
- `output/train_data.pkl` + `train_index.parquet`
- `output/val_data.pkl` + `val_index.parquet`  
- `output/test_data.pkl` + `test_index.parquet`

## 📊 Output Format

Each pickle file contains:

```python
{
    'data': [  # List of samples
        {
            'hadm_id': 12345,
            'subject_id': 67890,
            'admittime': Timestamp('2019-01-15 10:30:00'),
            'values': np.array([48, 120]),   # (48 hours, 120 features)
            'masks': np.array([48, 120]),    # Binary observation indicators
            'deltas': np.array([48, 120]),   # Hours since last observation
            'readmit_30d': 1,  # Label: 0 or 1
            'anchor_year_group': '2017 - 2019'
        },
        # ... more samples
    ],
    'feature_names': ['heart_rate', 'sbp', ...],  # List of feature names
    'global_medians': {'heart_rate': 80.5, ...},  # For reference
    'n_hours': 48,
    'n_features': 120
}
```

## 💻 Usage in Training

```python
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

# Load training data
with open('output/train_data.pkl', 'rb') as f:
    train_dict = pickle.load(f)

train_data = train_dict['data']
feature_names = train_dict['feature_names']

# PyTorch Dataset
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

# DataLoader
dataset = MIMICDataset(train_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in loader:
    values = batch['values']  # (32, 48, 120)
    masks = batch['masks']    # (32, 48, 120)
    deltas = batch['deltas']  # (32, 48, 120)
    labels = batch['label']   # (32, 1)
    
    # Your model forward pass
    # outputs = model(values, masks, deltas)
    # loss = criterion(outputs, labels)
```

## 🏗️ Pipeline Architecture

```
1. BigQuery (SQL)
   ├── 01_cohort_definition.sql → cohort.parquet
   ├── 02_extract_chartevents.sql → chartevents_raw.parquet
   ├── 03_extract_labevents.sql → labevents_raw.parquet
   └── 04_extract_prescriptions.sql → prescriptions_raw.parquet

2. Clean Units (Python)
   ├── Map itemids → feature names
   ├── Standardize units (°F→°C, etc.)
   ├── Remove duplicates
   └── Apply clinical bounds

3. Create Time Series (Python)
   ├── Bin to 48 1-hour windows
   ├── Aggregate (median) within bins
   └── Create (n_hours, n_features) matrix

4. Compute Features (Python)
   ├── Masks: 0/1 observation indicators
   ├── Deltas: Hours since last observed
   ├── Forward-fill: Carry forward last value
   └── Median imputation: Fill remaining NaNs

5. Temporal Split (Python)
   ├── Train: 2008-2013 (60%)
   ├── Val: 2014-2016 (20%)
   └── Test: 2017-2019 (20%)

6. Save Output (Python)
   ├── Pickle: Fast loading
   └── Parquet: Metadata queries
```

## 📈 Expected Statistics

```
Total admissions: ~195,000

Temporal split:
├── Train:      117,000 admissions (60%)
├── Validation:  39,000 admissions (20%)
└── Test:        39,000 admissions (20%)

Readmission rate: ~26.7%

Features: ~120
├── Chart events: ~80 (vitals, I/O, GCS)
├── Lab events:   ~35 (CBC, chemistry)
└── Medications:   ~5 (binary indicators)

Time series shape: (48 hours, 120 features)
Observation rate: ~15-30% (sparse, handled by masks/deltas)
```

## 🔧 Customization

### Adjust Time Window

Edit `config.yaml`:
```yaml
preprocessing:
  time_window_hours: 72  # Change from 48 to 72 hours
  bin_size_hours: 2      # Change from 1 to 2 hours → 36 bins
```

### Adjust Clinical Bounds

Edit `config.yaml`:
```yaml
clinical_bounds:
  heart_rate: [30, 200]  # Stricter bounds
  glucose: [50, 500]     # Custom range
```

### Change Temporal Split

Edit `config.yaml`:
```yaml
temporal_split:
  train_years: [2008, 2016]  # More training data
  val_years: [2016, 2018]
  test_years: [2018, 2020]
```

## 🎯 Model Integration Examples

### GRU-D (Decay)

```python
class GRUD(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.decay = nn.Linear(input_dim, input_dim)
        
    def forward(self, values, masks, deltas):
        batch_size, seq_len, feat_dim = values.shape
        h = torch.zeros(batch_size, self.hidden_dim)
        
        for t in range(seq_len):
            gamma = torch.exp(-F.relu(self.decay(deltas[:, t, :])))
            x_t = masks[:, t, :] * values[:, t, :] + \
                  (1 - masks[:, t, :]) * gamma * values[:, t, :]
            h = self.gru_cell(x_t, h)
        
        return h
```

### Transformer

```python
class TimeAwareTransformer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_embedding = nn.Linear(1, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=6
        )
        
    def forward(self, values, masks, deltas):
        x = self.embedding(values)
        t = self.time_embedding(deltas.mean(dim=-1, keepdim=True))
        x = x + t
        
        padding_mask = (masks.sum(dim=-1) == 0)
        x = x.permute(1, 0, 2)
        output = self.transformer(x, src_key_padding_mask=padding_mask)
        return output[-1]
```

## 🐛 Troubleshooting

### BigQuery Authentication Error
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Out of Memory
```yaml
# In config.yaml
preprocessing:
  chunk_size: 5000  # Reduce from 10000
```

### Feature Count Mismatch
Delete output directory and rerun:
```bash
rm -rf output/*
./run_all.sh
```

## 📝 Files Generated

```
output/
├── cohort.parquet                    # Cohort with labels
├── chartevents_raw.parquet          # Raw chart events
├── chartevents_clean.parquet        # Cleaned chart events
├── labevents_raw.parquet            # Raw lab events
├── labevents_clean.parquet          # Cleaned lab events
├── prescriptions_raw.parquet        # Raw prescriptions
├── prescriptions_clean.parquet      # Cleaned prescriptions
├── feature_names.txt                # List of all features
├── timeseries_binned.pkl            # After Step 3
├── timeseries_processed.pkl         # After Step 4
├── train_data.pkl ⭐                # FINAL: Training data
├── train_index.parquet ⭐           # FINAL: Training index
├── val_data.pkl ⭐                  # FINAL: Validation data
├── val_index.parquet ⭐             # FINAL: Validation index
├── test_data.pkl ⭐                 # FINAL: Test data
└── test_index.parquet ⭐            # FINAL: Test index
```

## ✅ Quality Checks

The pipeline automatically validates:
- ✓ No NaNs after imputation
- ✓ All masks in [0, 1]
- ✓ All deltas ≥ 0
- ✓ Correct shapes (48, n_features)
- ✓ No patient overlap between splits
- ✓ Labels in {0, 1}

## 📚 Next Steps

1. **Load data** using provided code snippets
2. **Explore features** using Parquet index
3. **Train your model** (GRU-D, Transformer, etc.)
4. **Evaluate** on held-out test set
5. **Iterate** - Adjust bounds, features, split ratios

## 📧 Questions?

See full documentation in `README.md` or create an issue on GitHub.

---

**Created**: November 10, 2025  
**Version**: 1.0.0  
**Authors**: Yuchen Zhou, Jiaqi Wu
