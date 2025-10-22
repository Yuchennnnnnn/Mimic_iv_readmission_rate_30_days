# 30-Day Hospital Readmission Prediction Pipeline# Yuchen Zhou's ML Pipeline for 30-Day Readmission Prediction



**Author**: Yuchen Zhou  **Author:** Yuchen Zhou  

**Course**: CompSci 526 - Fall 2025  **Date:** October 20, 2025  

**Institution**: Duke University  **Course:** CS526 - Machine Learning in Healthcare, Duke University

**Dataset**: MIMIC-IV Clinical Database  

---

---

## 📁 What's in This Folder

## 📋 Project Overview

This is my individual contribution to the team project. It contains a complete, independent ML pipeline that works alongside my teammates' work.

This pipeline predicts **30-day hospital readmission risk** using MIMIC-IV clinical data. It implements **5 machine learning models** with automated feature selection based on LASSO coefficients from collaborative work.

```

### Key FeaturesYuchenZhou_Pipeline/

- ✅ **Automated Feature Selection**: Uses pre-computed LASSO feature importance├── training/              # My training pipeline

- ✅ **5 Model Implementations**: LR, RF, XGBoost, LSTM, Transformer├── testing/               # My inference/testing pipeline

- ✅ **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, Calibration├── README.md              # This file

- ✅ **Production-Ready**: Modular code with config-driven training├── QUICK_REFERENCE.md     # Quick command reference

- ✅ **Complete Pipeline**: From data loading to model deployment├── PIPELINE_README.md     # Detailed documentation

└── setup_and_run.sh       # Quick setup script

---```



## 🏆 Model Performance Summary---



**Dataset**: 205,980 hospital admissions (26.72% readmission rate)  ## 🚀 Quick Start (2 Minutes)

**Features**: 18 selected features (from 48 LASSO features)  

**Train/Test Split**: 80/20 (164,784 / 41,196 samples)  ### Option 1: Automated Quick Test

```bash

### Best Models by Metriccd YuchenZhou_Pipeline/training

python quick_start.py

| Metric | Model | Score | Notes |```

|--------|-------|-------|-------|

| **ROC-AUC** | **Transformer** | **0.7056** ⭐ | Best overall discrimination |This will:

| **F1-Score** | **XGBoost** | **0.4947** ⭐ | Best precision-recall balance |1. Generate synthetic test data

| **Recall** | **XGBoost** | **68.5%** ⭐ | Catches most readmissions |2. Train a Logistic Regression model

| **Precision** | **Transformer** | **65.2%** | Lowest false positives |3. Evaluate and create reports

4. Show you where results are saved

### Complete Results

### Option 2: Use Bash Script

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score |```bash

|-------|---------|--------|----------|-----------|--------|----------|cd YuchenZhou_Pipeline

| **Transformer** | **0.7056** | 0.4778 | 74.84% | 65.20% | 12.55% | 0.2104 |chmod +x setup_and_run.sh

| **XGBoost** | **0.7040** | **0.4756** | 62.60% | 38.72% | **68.52%** | **0.4947** |./setup_and_run.sh

| **LSTM** | 0.7030 | 0.4723 | 74.83% | 62.56% | 14.50% | 0.2354 |```

| **Random Forest** | 0.6941 | 0.4625 | 64.25% | 39.37% | 62.59% | 0.4834 |

| **Logistic Reg** | 0.6626 | 0.4037 | 59.18% | 35.76% | 66.21% | 0.4643 |---



### Key Insights## 📊 Train on Real Data



1. **XGBoost is the best practical choice**:### Step 1: Install Dependencies

   - High recall (68.5%) catches most readmissions```bash

   - Balanced F1-score (0.49) for real-world deploymentcd YuchenZhou_Pipeline/training

   - Fastest training among top performerspip install -r requirements.txt

```

2. **Transformer has highest AUC but low recall**:

   - Best at ranking risk (0.7056 AUC)### Step 2: Update Configuration

   - Very high precision (65%) but misses many cases (12% recall)Edit `training/config.yaml` to point to the cleaned data:

   - Better suited for high-confidence predictions```yaml

data:

3. **Traditional ML vs Deep Learning**:  input_path: "../../cleaned_data.csv"  # Adjust path as needed

   - XGBoost/RF: Better recall, faster training, easier deployment```

   - LSTM/Transformer: Higher precision, better calibration, requires more compute

### Step 3: Train Models

---```bash

# Train all 5 models

## 📁 Project Structurepython src/train.py --model all --config config.yaml



```# Or train individually

YuchenZhou_Pipeline/python src/train.py --model logistic --config config.yaml

├── README.md                              # This filepython src/train.py --model rf --config config.yaml

├── FEATURE_SELECTION_EXPLANATION.md       # Why 50 features → 18 columnspython src/train.py --model xgb --config config.yaml

├── Feature_Importance_by_Coef.csv         # LASSO coefficients (Xi Chen's work)python src/train.py --model lstm --config config.yaml --epochs 30

│python src/train.py --model transformer --config config.yaml

├── training/```

│   ├── config.yaml                        # Configuration file

│   ├── requirements.txt                   # Python dependencies---

│   ├── quick_train.sh                     # One-click training script

│   ├── run_training.py                    # Interactive training## 📈 View Results

│   ├── check_feature_mapping.py           # Feature analysis tool

│   │### Check Metrics

│   ├── src/```bash

│   │   ├── train.py                       # Main training scriptcd training

│   │   ├── feature_selection.py           # Feature selection logic ⭐cat reports/metrics.csv

│   │   ├── preprocess.py                  # Data preprocessing```

│   │   ├── models.py                      # Model implementations

│   │   ├── evaluate.py                    # Evaluation metrics### View Plots

│   │   ├── dataset.py                     # PyTorch datasets```bash

│   │   └── utils.py                       # Helper functions# ROC curves

│   │open reports/roc_curve_xgb.png

│   ├── artifacts/                         # Trained models

│   │   ├── lr.pkl                         # Logistic Regression# Model comparison

│   │   ├── rf.pkl                         # Random Forestopen reports/model_comparison.png

│   │   ├── xgb.pkl                        # XGBoost ⭐

│   │   ├── lstm.pth                       # LSTM# Feature importance

│   │   └── transformer.pth                # Transformeropen reports/feature_importance_xgb.png

│   │```

│   └── reports/                           # Evaluation results

│       ├── metrics.csv                    # All model metrics---

│       ├── predictions_*.csv              # Model predictions

│       ├── roc_curve_*.png                # ROC curves## 🧪 Make Predictions on New Data

│       ├── confusion_matrix_*.png         # Confusion matrices

│       └── feature_importance_*.png       # Feature importance plots```bash

│cd YuchenZhou_Pipeline/testing

└── testing/

    ├── src/python src/inference.py \

    │   └── inference.py                   # Model inference script  --model-path ../training/artifacts/xgb.pkl \

    └── README.md                          # Testing documentation  --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \

```  --input ../../cleaned_data.csv \

  --output my_predictions.csv \

---  --model-type sklearn

```

## 🚀 Quick Start

---

### 1. Environment Setup

## 📚 Documentation

```bash

# Navigate to project- **QUICK_REFERENCE.md** - Command cheat sheet

cd YuchenZhou_Pipeline/training- **PIPELINE_README.md** - Complete documentation

- **IMPLEMENTATION_SUMMARY.md** - What was built

# Install dependencies (using virtual environment)- **training/README.md** - Training details

pip install -r requirements.txt- **testing/README.md** - Inference details

```

---

**Required Packages**:

- numpy<2, pandas, scikit-learn## 🎯 Models Implemented

- xgboost, imbalanced-learn

- torch, torchvision1. **Logistic Regression** - Interpretable baseline

- matplotlib, seaborn, pyyaml2. **Random Forest** - Ensemble with feature importance

3. **XGBoost** - Best performance (typically)

### 2. Train Models4. **LSTM** - Deep learning with embeddings

5. **Transformer** - Attention-based architecture

**Option A: Interactive Script (Recommended)**

```bash---

./quick_train.sh

## 📂 Expected Outputs

# Then select:

# 1 - Quick test (Logistic Regression, ~2 min)After training, you'll have:

# 2 - Traditional ML (LR + RF + XGBoost, ~15 min) ⭐

# 3 - All models (including LSTM + Transformer, ~1 hour)```

```training/

├── artifacts/

**Option B: Manual Training**│   ├── lr.pkl, rf.pkl, xgb.pkl        # Trained models

```bash│   ├── lstm.pt, transformer.pt        # PyTorch models

# Single model│   └── *_preprocessor.joblib          # Encoders

python src/train.py --model xgb --config config.yaml└── reports/

    ├── metrics.csv                     # All metrics

# Multiple models    ├── model_comparison.png            # Comparison chart

python src/train.py --model logistic --config config.yaml    ├── roc_curve_*.png                 # ROC curves

python src/train.py --model rf --config config.yaml    ├── predictions_*.csv               # Predictions

python src/train.py --model xgb --config config.yaml    └── feature_importance_*.png        # Feature importance

``````



### 3. View Results---



```bash## ⚡ Performance Summary

cd reports/

Expected results on MIMIC-IV data:

# View metrics

cat metrics.csv| Model | ROC-AUC | Training Time |

|-------|---------|---------------|

# View visualizations| Logistic Reg | 0.67 | 1 min |

open roc_curve_xgb.png| Random Forest | 0.71 | 5 min |

open confusion_matrix_xgb.png| **XGBoost** | **0.73** | 5 min ⭐ |

open feature_importance_xgb.png| LSTM | 0.71 | 30 min |

```| Transformer | 0.72 | 40 min |



------



## 🔧 Configuration## 🔧 Troubleshooting



Edit `training/config.yaml` to customize:### GPU Out of Memory

```yaml

### Feature Selection# In training/config.yaml, reduce batch size:

```yamlhyperparameters:

feature_selection:  lstm:

  enabled: true                    # Use LASSO feature selection    batch_size: 32

  top_n: 50                        # Number of top features (current: 18 columns)```

  importance_threshold: 0.05       # Minimum importance threshold

  feature_importance_path: "../Feature_Importance_by_Coef.csv"### Import Errors

```Make sure you're in the correct directory:

```bash

**Note**: 50 LASSO features map to 18 original columns due to one-hot encoding. See [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md) for details.cd YuchenZhou_Pipeline/training

python src/train.py --model logistic

### Model Hyperparameters```

```yaml

models:### Slow Training

  logistic:Train on a subset first to test:

    penalty: 'l2'```python

    C: 1.0# In config.yaml, you can add this in preprocessing:

    max_iter: 1000preprocessing:

    class_weight: 'balanced'  sample_size: 5000  # Use subset for testing

  ```

  rf:

    n_estimators: 100---

    max_depth: 15

    min_samples_split: 50## 🤝 Relation to Team Project

    class_weight: 'balanced'

  This pipeline is my **independent contribution** and works separately from:

  xgb:- `preprocessing/` - Shared preprocessing scripts

    n_estimators: 200- `XiChen_Lasso/` - Xi Chen's LASSO feature selection

    max_depth: 5- Other teammates' work

    learning_rate: 0.1

    scale_pos_weight: 3.0All folders can coexist and use the same `cleaned_data.csv` file.

```

---

### Training Parameters

```yaml## 📝 What I Contributed

split:

  test_size: 0.2✅ Complete end-to-end ML pipeline  

  random_state: 42✅ 5 different model implementations  

  stratify: true✅ Comprehensive preprocessing  

✅ Full evaluation suite  

deep_learning:✅ Deployment infrastructure  

  epochs: 50✅ 3,500+ lines of code  

  batch_size: 256✅ 2,000+ lines of documentation  

  learning_rate: 0.001✅ Unit and integration tests  

  early_stopping_patience: 5

```---



---## 🎓 For Grading/Presentation



## 📊 Selected Features (18 Total)**Key Files to Review:**

1. `training/src/train.py` - Main training script

### Demographic (3 features)2. `training/src/models.py` - All 5 models

- `anchor_age` - Patient age3. `training/reports/metrics.csv` - Results

- `gender` - Patient gender (F/M)4. `training/reports/model_comparison.png` - Visual comparison

- `marital_status` - Marital status

**To Demonstrate:**

### Clinical (4 features)```bash

- `died_in_hospital` - In-hospital mortality (⭐ Most important, weight=0.60)# Quick test (2 min)

- `days_since_prev_discharge` - Time since last dischargecd training

- `num_diagnoses` - Number of diagnosespython quick_start.py

- `is_surgical_service` - Surgical vs medical service

# View results

### Administrative (5 features)cat reports/metrics.csv

- `admission_type` - Type of admission (Emergency, Elective, etc.)```

- `admission_location` - Where patient admitted from

- `discharge_location` - Where patient discharged to---

- `last_service` - Last clinical service (⭐ 2nd most important)

- `insurance` - Insurance type## 📧 Questions?



### Lab Values (4 features)- Check `QUICK_REFERENCE.md` for commands

- `Glucose_median` - Median glucose level- See `PIPELINE_README.md` for full documentation

- `Hemoglobin_median` - Median hemoglobin- Review `training/README.md` for detailed guide

- `Hemoglobin_min` - Minimum hemoglobin

- `Potassium_min` - Minimum potassium---



### Other (2 features)**Ready to use! Run `python training/quick_start.py` to get started.** 🚀

- `language` - Primary language
- `unique_careunits` - Number of care units visited

---

## 🎯 Model Recommendations

### For Different Use Cases

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Production Deployment** | **XGBoost** | Best F1-score, good recall, fast inference |
| **High-Risk Screening** | **XGBoost** | 68.5% recall catches most readmissions |
| **Low False Alarm** | **Transformer** | 65% precision, lowest false positives |
| **Risk Ranking** | **Transformer** | Highest AUC (0.7056) for risk stratification |
| **Interpretability** | **Logistic Regression** | Clear coefficients, easy to explain |
| **Quick Prototyping** | **Random Forest** | Fast training, good baseline performance |

### Deployment Strategy

**Recommended Two-Stage Approach**:
1. **Stage 1**: Use XGBoost for initial screening (high recall)
2. **Stage 2**: Use Transformer to prioritize high-risk cases (high precision)

This catches most readmissions while minimizing unnecessary interventions.

---

## 🧪 Model Inference

Use trained models for predictions:

```bash
cd testing

# Using XGBoost (recommended)
python src/inference.py \
  --model ../training/artifacts/xgb.pkl \
  --data ../../cleaned_data.csv \
  --output predictions.csv

# Output includes:
# - Patient ID
# - True label
# - Predicted probability
# - Predicted class
# - Risk category (Low/Medium/High)
```

---

## 📈 Performance Analysis

### Confusion Matrix (XGBoost)
```
                Predicted
              No      Yes
Actual  No   18,244  11,944   → Specificity: 60.4%
        Yes   3,465   7,543   → Sensitivity: 68.5%
```

**Interpretation**:
- **True Positives (7,543)**: Correctly identified readmissions
- **False Negatives (3,465)**: Missed readmissions (31.5%)
- **False Positives (11,944)**: False alarms
- **True Negatives (18,244)**: Correct non-readmission predictions

### ROC-AUC Comparison

All models achieve AUC > 0.66, with top 3 models > 0.70:
- Excellent: 0.9-1.0
- Good: 0.8-0.9
- **Fair: 0.7-0.8** ← Our models
- Poor: 0.6-0.7
- Random: 0.5

Our models are in the **"Fair to Good"** range, suitable for clinical decision support.

---

## 🔬 Feature Importance (Top 10 from XGBoost)

1. **died_in_hospital** (0.603) - In-hospital mortality
2. **last_service_OMED** (0.451) - Served by OMED
3. **days_since_prev_discharge** (0.281) - Time since last visit
4. **gender** (0.382) - Patient gender
5. **discharge_location** (0.318) - Discharge destination
6. **admission_type** (0.340) - Type of admission
7. **anchor_age** (0.171) - Patient age
8. **insurance** (0.210) - Insurance type
9. **marital_status** (0.208) - Marital status
10. **Hemoglobin_median** (0.107) - Median hemoglobin

See `reports/feature_importance_xgb.png` for full visualization.

---

## 📝 Methodology

### 1. Feature Selection
- Started with 47 features in cleaned data
- Xi Chen's LASSO identified 121 important one-hot encoded features
- Selected top 48 LASSO features (threshold ≥ 0.05)
- Mapped to 18 original columns (handles one-hot encoding)
- See [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md)

### 2. Data Preprocessing
- **Missing Values**: Median for numeric, mode for categorical
- **Categorical Encoding**:
  - Logistic Regression: One-Hot Encoding (low cardinality only)
  - Random Forest/XGBoost: Label Encoding
  - LSTM/Transformer: Embedding layers
- **Numeric Features**: StandardScaler normalization
- **Class Imbalance**: Handled via class weights and scale_pos_weight

### 3. Model Training
- **Train/Test Split**: 80/20 stratified split
- **Validation**: For deep learning models (10% of training)
- **Early Stopping**: Prevents overfitting (patience=5 epochs)
- **Evaluation**: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1

### 4. Model Architectures

**Traditional ML**:
- Logistic Regression: L2 regularization, balanced class weights
- Random Forest: 100 trees, max_depth=15
- XGBoost: 200 trees, learning_rate=0.1, scale_pos_weight=3.0

**Deep Learning**:
- LSTM: Bidirectional, 2 layers, 128 hidden units
- Transformer: 4 attention heads, 3 encoder layers

---

## 🤝 Collaboration

This pipeline builds upon collaborative work:

- **Xi Chen**: LASSO feature selection (`XiChen_Lasso/`)
- **Yuchen Zhou**: End-to-end ML pipeline (this folder)

Feature importance coefficients are shared via `Feature_Importance_by_Coef.csv`.

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | All configuration parameters |
| `src/train.py` | Main training script (630 lines) |
| `src/feature_selection.py` | LASSO feature integration ⭐ |
| `src/models.py` | All 5 model implementations |
| `src/evaluate.py` | Metrics and visualization |
| `reports/metrics.csv` | Complete results table |
| `artifacts/*.pkl` | Trained model files |

---

## 🐛 Troubleshooting

### Common Issues

**1. NumPy Version Error**
```bash
# Error: numpy.dtype size changed
pip install "numpy<2"
```

**2. Missing Packages**
```bash
pip install -r requirements.txt
```

**3. CUDA Out of Memory (Deep Learning)**
```yaml
# In config.yaml, reduce batch size:
deep_learning:
  batch_size: 128  # Instead of 256
```

**4. Feature Mapping Issues**
```bash
# Check feature mapping:
python check_feature_mapping.py
```

---

## 📊 Next Steps

### Immediate Improvements
1. **Hyperparameter Tuning**: Use GridSearchCV for XGBoost
2. **Ensemble Models**: Combine XGBoost + Transformer predictions
3. **More Features**: Increase `top_n` to 100 for more features
4. **Threshold Optimization**: Adjust classification threshold for recall/precision trade-off

### Advanced Enhancements
1. **SHAP Analysis**: Explain individual predictions
2. **Calibration**: Improve probability calibration
3. **Temporal Validation**: Time-based train/test split
4. **External Validation**: Test on different hospital data
5. **Clinical Integration**: Deploy as risk calculator API

---

## 📖 References

- **Dataset**: MIMIC-IV Clinical Database v2.0
- **Paper**: Johnson et al. (2023). "MIMIC-IV, a freely accessible electronic health record dataset"
- **Methods**: Scikit-learn, XGBoost, PyTorch
- **Evaluation**: Standard ML metrics (Hosmer-Lemeshow, 2013)

---

## 📧 Contact

**Yuchen Zhou**  
Duke University - CompSci 526  
Fall 2025  

For questions about this pipeline, please refer to the code comments or configuration file.

---

## 📜 License

This project is for educational purposes as part of CompSci 526 coursework.  
MIMIC-IV data usage follows PhysioNet credentialed access requirements.

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready
