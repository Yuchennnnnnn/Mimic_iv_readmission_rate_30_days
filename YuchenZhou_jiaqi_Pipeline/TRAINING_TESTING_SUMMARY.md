# Training & Testing Data Summary Report

**Author**: Yuchen Zhou  
**Project**: 30-Day Readmission Prediction Pipeline  
**Date**: October 2025  

---

## 📊 Dataset Basic Information

- **Original Dataset**: 205,980 samples
- **Number of Features**: 47 columns (original) → 18 columns (after feature selection)
- **Readmission Rate**: 26.72% (55,039 / 205,980)
- **No Readmission**: 73.28% (150,941 / 205,980)

---

## 🎯 TRAINING Results Summary (20% Test Set)

### Training Configuration

- **Training Set**: 164,784 samples (80%)
- **Test Set**: 41,196 samples (20%)
- **Number of Features**: 18 (mapped from 48 LASSO features)
- **Trained Models**: 5 (LR, RF, XGBoost, LSTM, Transformer)

### Model Performance Comparison (on 20% test set)

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|------|---------|----------|-----------|--------|----------|
| **Transformer** | **0.7056** ⭐ | 74.84% | 65.20% | 12.55% | 0.2104 |
| **XGBoost** | **0.7040** | 62.60% | 38.71% | **68.52%** ⭐ | **0.4947** ⭐ |
| **LSTM** | 0.7030 | 74.83% | 62.56% | 14.50% | 0.2354 |
| **Random Forest** | 0.6941 | 64.25% | 39.37% | 62.59% | 0.4834 |
| **Logistic Reg** | 0.6626 | 59.18% | 35.76% | 66.21% | 0.4643 |

### 🏆 Best Models

- **Highest ROC-AUC**: Transformer (0.7056) - Best risk ranking capability
- **Highest F1-Score**: XGBoost (0.4947) - Best precision-recall balance
- **Highest Recall**: XGBoost (68.52%) - Captures most readmission patients

### Confusion Matrix Details (test set 41,196 samples)

#### Logistic Regression
```
                Predicted
              No Readm  Readm
Actual No Readm   17,093   13,095  ← Specificity: 56.62%
       Readm       3,720    7,288  ← Sensitivity: 66.21%
```

#### Random Forest
```
                Predicted
              No Readm  Readm
Actual No Readm   19,579   10,609  ← Specificity: 64.86%
       Readm       4,118    6,890  ← Sensitivity: 62.59%
```

#### XGBoost ⭐ (Recommended)
```
                Predicted
              No Readm  Readm
Actual No Readm   18,244   11,944  ← Specificity: 60.43%
       Readm       3,465    7,543  ← Sensitivity: 68.52%
```
- **Correctly identified**: 7,543 patients who will be readmitted
- **Missed**: 3,465 patients who will be readmitted (31.5%)
- **False alarms**: 11,944 patients who will not be readmitted

#### LSTM
```
                Predicted
              No Readm  Readm
Actual No Readm   29,233      955  ← Specificity: 96.84%
       Readm       9,412    1,596  ← Sensitivity: 14.50%
```
- **Characteristics**: Very high specificity, very low sensitivity (too conservative)

#### Transformer
```
                Predicted
              No Readm  Readm
Actual No Readm   29,451      737  ← Specificity: 97.56%
       Readm       9,627    1,381  ← Sensitivity: 12.55%
```
- **Characteristics**: Highest specificity and precision, but lowest recall

### Training Generated Files

**Location**: `training/reports/`

| File Type | Count | Description |
|---------|------|------|
| `metrics.csv` | 1 | Performance metrics summary for all models |
| `predictions_*.csv` | 5 | Prediction results for each model on test set (41,196 samples) |
| `roc_curve_*.png` | 5 | ROC curve visualization |
| `pr_curve_*.png` | 5 | Precision-Recall curve |
| `confusion_matrix_*.png` | 5 | Confusion matrix visualization |
| `calibration_curve_*.png` | 5 | Probability calibration curve |
| `feature_importance_*.png` | 3 | Feature importance (LR, RF, XGB) |

---

## 🧪 TESTING/INFERENCE Results Summary (Full Dataset)

### Inference Configuration

- **Inference Data**: Full dataset (205,980 samples)
- **Purpose**: Generate readmission risk predictions for all patients
- **Completed Models**: XGBoost

### Inference Results (XGBoost)

- **Total Samples**: 205,980
- **Predicted Readmission**: 97,356 cases (47.3%)
- **Average Prediction Probability**: 0.486
- **Probability Range**: [0.001, 0.981]

**Interpretation**:
- XGBoost model predicts 47.3% of patients are at readmission risk
- Actual readmission rate is 26.72%
- Model tends to prefer false alarms over missing cases (suitable for clinical intervention)

### Testing Generated Files

**Location**: `testing/reports/`

| File | Sample Count | Description |
|------|--------|------|
| `predictions_xgb.csv` | 205,980 | XGBoost full dataset predictions |

**Pending**: LR, RF full dataset inference

---

## 📖 Training vs Testing Key Differences

### 🎯 TRAINING (training/reports/)

| Dimension | Details |
|------|------|
| **Data** | 20% test set (41,196 samples) |
| **Purpose** | Evaluate model performance, calculate metrics |
| **Contains** | True labels + predictions |
| **Output Metrics** | ROC-AUC, Precision, Recall, F1, Confusion Matrix |
| **File Content** | `predictions_*.csv` includes true label comparison |
| **Use Case** | Model selection, performance evaluation, paper writing |

**predictions_*.csv Format Example**:
```csv
subject_id,hadm_id,true_label,predicted_label,predicted_probability
10000032,22841357,1,1,0.524
10000032,29079034,0,1,0.616
...
```

### 🧪 TESTING/INFERENCE (testing/reports/)

| Dimension | Details |
|------|------|
| **Data** | Full dataset (205,980 samples) |
| **Purpose** | Generate risk predictions for all patients |
| **Contains** | Prediction probabilities + predicted labels |
| **Output** | Readmission risk score for each patient |
| **File Content** | `predictions_*.csv` for practical application |
| **Use Case** | Clinical decision support, risk stratification, intervention priority |

**predictions_*.csv Format Example**:
```csv
subject_id,hadm_id,predicted_label,predicted_probability
10000032,22841357,1,0.524
10000032,29079034,1,0.616
...
```

---

## 📁 Complete File Structure

```
YuchenZhou_Pipeline/
│
├── training/
│   ├── artifacts/                    # Trained models
│   │   ├── lr.pkl                   # Logistic Regression
│   │   ├── rf.pkl                   # Random Forest
│   │   ├── xgb.pkl                  # XGBoost ⭐
│   │   ├── lstm.pt                  # LSTM
│   │   ├── transformer.pt           # Transformer
│   │   └── *_preprocessor.joblib    # Corresponding preprocessors
│   │
│   └── reports/                      # Training results
│       ├── metrics.csv              # Performance metrics (41,196 samples)
│       ├── predictions_*.csv        # Test set predictions (5 models)
│       ├── roc_curve_*.png          # ROC curves (5)
│       ├── confusion_matrix_*.png   # Confusion matrices (5)
│       └── feature_importance_*.png # Feature importance (3)
│
└── testing/
    └── reports/                      # Testing/Inference results
        └── predictions_xgb.csv      # Full dataset predictions (205,980 samples)
```

---

## 💡 Key Findings

### 1. Model Performance

- **XGBoost is the best production model**: 
  - High recall (68.5%) suitable for capturing readmission patients
  - Balanced F1 score (0.49) suitable for practical application
  - Fast training, good interpretability

- **Transformer has highest AUC but low recall**:
  - Good for risk ranking, not for binary classification
  - High precision (65%) but many missed cases (87.5%)
  - High computational cost

### 2. Clinical Application Recommendations

**Recommended Strategy**: Two-stage approach
1. **Stage 1**: Use XGBoost for initial screening (high recall)
2. **Stage 2**: Use Transformer for refined ranking (high precision)

This captures most readmission patients while reducing unnecessary interventions.

### 3. Feature Selection Effectiveness

- **Dimensionality reduction**: From 47 columns → 18 columns (reduced by 61.7%)
- **Performance**: AUC reaches 0.70+, excellent results
- **Most important features**:
  1. died_in_hospital (0.603)
  2. last_service_OMED (0.451)
  3. gender (0.382)

---

## 🔧 How to Use These Results

### For Paper Writing

```
We trained 5 models on the MIMIC-IV dataset (205,980 hospitalization records).
Using 80/20 stratified split, we evaluated performance on 41,196 test samples.
XGBoost achieved the best overall performance (AUC=0.7040, Recall=68.5%, F1=0.4947),
correctly identifying 68.5% of patients who will be readmitted.
```

### For Clinical Reporting

```
The model achieved 62.6% accuracy on the test set. For patients who will be readmitted,
it correctly identifies 68.5% (7,543/11,008). The false alarm rate is about 39.6%,
which is acceptable in clinical intervention scenarios (prefer more intervention over missing cases).
```

### Next Steps

1. ✅ **Complete inference for other models** - Run `./run_all_inference.sh`
2. ✅ **Hyperparameter tuning** - GridSearch to optimize XGBoost
3. ✅ **Feature expansion** - Try top_n: 100
4. ✅ **Model ensemble** - Combine predictions from multiple models

---

## 📊 Quick Commands

### View Training Results
```bash
cd YuchenZhou_Pipeline/training/reports
cat metrics.csv
open roc_curve_xgb.png
open confusion_matrix_xgb.png
```

### Run Complete Testing
```bash
cd YuchenZhou_Pipeline/testing
./run_all_inference.sh
```

### Regenerate Report
```bash
cd YuchenZhou_Pipeline
python generate_summary.py
```

---

**Last Updated**: October 2025  
**Status**: ✅ Training completed, Testing partially completed  
**Recommended Model**: XGBoost (AUC=0.7040, F1=0.4947)
