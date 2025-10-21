# 30-Day Readmission Prediction Pipeline

A comprehensive, reproducible end-to-end machine learning pipeline for predicting 30-day hospital readmissions using MIMIC-IV data.

## 📋 Overview

This pipeline implements multiple state-of-the-art models for binary classification:
- **Logistic Regression** - Linear baseline with L2 regularization
- **Random Forest** - Ensemble tree-based model
- **XGBoost** - Gradient boosting with advanced regularization
- **LSTM** - Recurrent neural network with bidirectional architecture
- **Transformer** - Self-attention mechanism (TabTransformer variant)

All models use proper preprocessing, handle imbalanced data, and produce comprehensive evaluation reports.

---

## 🗂 Project Structure

```
training/
├── src/
│   ├── preprocess.py      # Data loading, encoding, feature engineering
│   ├── models.py          # Model definitions (sklearn + PyTorch)
│   ├── train.py           # Main training CLI script
│   ├── evaluate.py        # Metrics computation and visualization
│   ├── dataset.py         # PyTorch Dataset classes
│   └── utils.py           # Helper functions
├── artifacts/             # Saved models and encoders
├── reports/               # Evaluation metrics and plots
├── data/                  # Processed datasets (optional)
├── tests/                 # Unit tests
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- pandas, numpy, scikit-learn, xgboost
- torch, torchvision
- matplotlib, seaborn, tqdm, pyyaml

### 2. Prepare Data

Ensure your data file is accessible and update `config.yaml`:

```yaml
data:
  input_path: "../cleaned_data.csv"  # Path to your data
```

**Required columns:**
- `subject_id`, `hadm_id` (identifiers)
- `readmit_label` (target: 0/1)
- Categorical columns (e.g., `gender`, `insurance`, `discharge_location`)
- Numeric columns (e.g., `age`, `length_of_stay`, `num_diagnoses`)

### 3. Train Models

**Train all models:**
```bash
python src/train.py --model all --config config.yaml
```

**Train specific model:**
```bash
# Logistic Regression
python src/train.py --model logistic --config config.yaml

# Random Forest
python src/train.py --model rf --config config.yaml

# XGBoost
python src/train.py --model xgb --config config.yaml

# LSTM
python src/train.py --model lstm --config config.yaml

# Transformer
python src/train.py --model transformer --config config.yaml
```

**Override hyperparameters:**
```bash
python src/train.py --model lstm --epochs 30 --batch-size 128 --learning-rate 0.001
```

### 4. View Results

After training, check the `reports/` directory:

```
reports/
├── metrics.csv                          # All model metrics
├── model_comparison.png                 # Side-by-side comparison
├── predictions_logistic.csv             # Per-model predictions
├── roc_curve_logistic.png              # ROC curves
├── pr_curve_logistic.png               # Precision-Recall curves
├── confusion_matrix_logistic.png       # Confusion matrices
├── calibration_curve_logistic.png      # Calibration plots
└── feature_importance_logistic.png     # Feature importance (if available)
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Data Configuration
```yaml
data:
  input_path: "../cleaned_data.csv"
  output_dir: "artifacts"
  reports_dir: "reports"
```

### Column Definitions
```yaml
columns:
  id_cols: ["subject_id", "hadm_id"]
  time_cols: ["admittime", "dischtime"]
  label: "readmit_label"
  categorical_cols:
    - "gender"
    - "insurance"
    - "admission_type"
    # ... add your categorical columns
  numeric_cols:
    - "anchor_age"
    - "length_of_stay"
    - "num_diagnoses"
    # ... add your numeric columns
```

### Model Hyperparameters
```yaml
hyperparameters:
  logistic:
    penalty: "l2"
    C: 1.0
    max_iter: 1000
    class_weight: "balanced"
  
  rf:
    n_estimators: 100
    max_depth: 10
    class_weight: "balanced"
  
  xgb:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 3.0
  
  lstm:
    hidden_dim: 128
    num_layers: 2
    bidirectional: true
    dropout: 0.3
    batch_size: 64
    num_epochs: 50
    learning_rate: 0.001
    early_stopping_patience: 5
  
  transformer:
    d_model: 128
    nhead: 8
    num_layers: 3
    dim_feedforward: 512
    dropout: 0.3
    batch_size: 64
    num_epochs: 50
    learning_rate: 0.0001
    early_stopping_patience: 5
```

---

## 🔍 How It Works

### Preprocessing Pipeline

#### **For Logistic Regression:**
- **Categorical:** OneHotEncoder for low-cardinality features (≤10 unique values)
- **Numeric:** StandardScaler (z-score normalization)
- High-cardinality categoricals are dropped to avoid curse of dimensionality

#### **For Random Forest:**
- **Categorical:** OrdinalEncoder (integer encoding)
- **Numeric:** Passthrough (no scaling needed for tree models)

#### **For XGBoost:**
- **Categorical:** OrdinalEncoder with unknown value handling
- **Numeric:** Passthrough
- Optional: Native categorical support with `enable_categorical=True`

#### **For LSTM / Transformer:**
- **Categorical:** Label encoding to contiguous integers (0=PAD, 1=UNK, 2+=categories)
- **Embedding dimensions:** Calculated using heuristic: `min(50, 1.6 × vocab_size^0.56)`
- **Numeric:** StandardScaler
- Creates PyTorch tensors with proper padding and attention masks

### Model Architectures

#### **LSTM**
- Multiple embedding layers (one per categorical feature)
- Bidirectional LSTM with 2 layers
- Dropout regularization
- Takes last hidden state → Linear → Sigmoid

#### **Transformer (TabTransformer)**
- Embeddings projected to unified dimension (`d_model`)
- Multi-head self-attention (8 heads)
- Column embeddings (learnable positional encodings)
- MLP for final prediction combining categorical and continuous features

### Training Strategy

**Sklearn Models:**
- Fit on training set with class balancing (`class_weight='balanced'`)
- Direct prediction on test set
- Support for cross-validation (configurable)

**PyTorch Models:**
- 85/15 train/validation split
- Adam optimizer with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on validation AUC
- Gradient clipping (max norm = 1.0)
- BCEWithLogitsLoss for numerical stability

### Evaluation Metrics

All models are evaluated using:
- **ROC-AUC** - Area under ROC curve
- **PR-AUC** - Area under Precision-Recall curve (better for imbalanced data)
- **F1-Score** - Harmonic mean of precision and recall
- **Accuracy** - Overall correctness
- **Precision** - True positive rate among predicted positives
- **Recall (Sensitivity)** - True positive rate among actual positives
- **Specificity** - True negative rate among actual negatives
- **Confusion Matrix** - TP, FP, TN, FN breakdown

---

## 📊 Expected Performance

Typical results on MIMIC-IV readmission data:

| Model          | ROC-AUC | PR-AUC | F1    | Accuracy |
|----------------|---------|--------|-------|----------|
| Logistic Reg.  | 0.65-0.70 | 0.30-0.35 | 0.35-0.40 | 0.60-0.65 |
| Random Forest  | 0.68-0.73 | 0.32-0.38 | 0.38-0.43 | 0.62-0.67 |
| XGBoost        | 0.70-0.75 | 0.35-0.40 | 0.40-0.45 | 0.65-0.70 |
| LSTM           | 0.68-0.74 | 0.33-0.39 | 0.38-0.44 | 0.63-0.68 |
| Transformer    | 0.69-0.75 | 0.34-0.40 | 0.39-0.45 | 0.64-0.69 |

*Note: Exact results depend on data quality, preprocessing, and hyperparameters*

---

## 🧪 Testing

Run unit tests to verify correctness:

```bash
cd training
pytest tests/ -v
```

**Test coverage:**
- ✅ Data loading and validation
- ✅ Column type detection
- ✅ Missing value handling
- ✅ Encoder roundtrip (encode → decode)
- ✅ Integration test with synthetic data

---

## 📝 Reproducibility Checklist

✅ **Seeds set** - All random operations use fixed seed (default: 42)  
✅ **Deterministic mode** - PyTorch deterministic algorithms enabled  
✅ **Saved artifacts** - All encoders, mappings, and models saved  
✅ **Versioned data** - Train/test split indices can be saved  
✅ **Logged hyperparameters** - All configs saved with results  
✅ **Evaluation scripts** - Automated metric computation and visualization

---

## 🔧 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size in config.yaml:
```yaml
hyperparameters:
  lstm:
    batch_size: 32  # Reduce from 64
  transformer:
    batch_size: 32
```

### Issue: High-cardinality categorical columns
**Solution:** Increase threshold or use feature engineering:
```yaml
preprocessing:
  ohe_cardinality_threshold: 20  # Increase from 10
  rare_category_threshold: 50    # Group rare categories
```

### Issue: Class imbalance
**Solution:** Adjust class weights or use SMOTE:
```yaml
hyperparameters:
  xgb:
    scale_pos_weight: 5.0  # Increase for more imbalanced data
```

### Issue: Overfitting in deep models
**Solution:** Increase dropout and weight decay:
```yaml
hyperparameters:
  lstm:
    dropout: 0.5  # Increase from 0.3
    weight_decay: 0.001  # Increase from 0.0001
```

---

## 📚 Advanced Usage

### Custom Feature Engineering

Edit `preprocess.py` to add custom features:

```python
def create_custom_features(df):
    """Add domain-specific features."""
    # Example: Interaction features
    df['age_los_interaction'] = df['anchor_age'] * df['length_of_stay']
    
    # Example: Binning
    df['age_group'] = pd.cut(df['anchor_age'], bins=[0, 40, 60, 100], 
                             labels=['young', 'middle', 'senior'])
    return df
```

### Hyperparameter Tuning

Use scikit-learn's GridSearchCV for sklearn models:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

### Using Saved Models for Inference

```python
import joblib
import torch
from src import preprocess, models

# Load sklearn model
model = joblib.load('artifacts/rf.pkl')
preprocessor = joblib.load('artifacts/rf_preprocessor.joblib')

# Transform new data
X_new_transformed = preprocessor.transform(X_new)
predictions = model.predict_proba(X_new_transformed)[:, 1]

# Load PyTorch model
config = load_config('config.yaml')
vocab_sizes = json.load(open('artifacts/lstm_vocab_sizes.json'))
embedding_dims = json.load(open('artifacts/lstm_embedding_dims.json'))

model = models.create_lstm_model(config, vocab_sizes, embedding_dims, continuous_dim)
model.load_state_dict(torch.load('artifacts/lstm.pt'))
model.eval()
```

---

## 🤝 Contributing

To add a new model:

1. Implement model class in `src/models.py`
2. Add preprocessing logic in `src/preprocess.py`
3. Add training logic in `src/train.py`
4. Update `config.yaml` with hyperparameters
5. Add tests in `tests/`

---

## 📄 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{readmission_pipeline_2025,
  title={End-to-End Readmission Prediction Pipeline},
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/readmission-pipeline}
}
```

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

---

## 🎯 Future Enhancements

- [ ] Add interpretability tools (SHAP, LIME)
- [ ] Implement time-series LSTM for sequential visits
- [ ] Add multi-task learning (readmission + mortality)
- [ ] Support for federated learning
- [ ] Automated hyperparameter tuning with Optuna
- [ ] Model ensembling (stacking, voting)
- [ ] Deployment scripts (Flask API, Docker)

---

**Happy Modeling! 🚀**
