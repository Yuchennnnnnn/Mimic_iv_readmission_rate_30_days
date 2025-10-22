# Training Results Summary

## 🎯 Training Configuration

### Dataset
- **Data File**: `cleaned_data.csv`
- **Total Samples**: 205,980
- **Training Set**: 164,784 samples (80%)
- **Test Set**: 41,196 samples (20%)
- **Readmission Rate**: 26.72%

### Feature Selection
✅ **Feature Selection Enabled** (Based on Xi Chen's LASSO results)

- **Original Features**: 47 columns
- **LASSO Selected Features**: 121 features (from `Feature_Importance_by_Coef.csv`)
- **Top-N Selection**: 50 most important features
- **Importance Threshold**: ≥ 0.05
- **Final Features Used**: 18 features

#### Top 5 Most Important Features:
1. **died_in_hospital** (0.6032) - In-hospital death
2. **last_service_OMED** (0.4505) - Last service type
3. **gender_F** (0.3823) - Gender (Female)
4. **admission_type_SURGICAL SAME DAY ADMISSION** (0.3403) - Admission type
5. **discharge_location_HOSPICE** (0.3182) - Discharge location

---

## 📊 Model Performance - Logistic Regression

### Training Details
- **Model Type**: Logistic Regression (L2 regularization)
- **Feature Dimension**: 30 (including One-Hot encoded features)
- **Training Time**: ~2 minutes

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.6626 |
| **PR-AUC** | 0.4037 |
| **Accuracy** | 0.5918 (59.18%) |
| **Precision** | 0.3576 (35.76%) |
| **Recall** | 0.6621 (66.21%) |
| **F1-Score** | 0.4643 (46.43%) |
| **Specificity** | 0.5662 (56.62%) |

### Confusion Matrix

|           | Predicted: No Readmission | Predicted: Readmission |
|-----------|---------------------------|------------------------|
| **Actual: No Readmission** | 17,093 (TN) | 13,095 (FP) |
| **Actual: Readmission**   | 3,720 (FN)  | 7,288 (TP)  |

### Performance Analysis

**Strengths** ✅:
- High recall (66.21%) - Can identify most patients who will actually be readmitted
- ROC-AUC = 0.66 - Significant improvement over random guessing (0.5)
- Effective feature selection - Reasonable performance with only 18 features

**Areas for Improvement** ⚠️:
- Low precision (35.76%) - Many false positives
- F1-Score = 0.46 - Imbalance between precision and recall
- Class imbalance issue (73% vs 27%)

---

## 📁 Generated Files

### Model Files (`artifacts/`)
- `lr.pkl` - Trained Logistic Regression model
- `lr_encoders.pkl` - One-Hot encoders
- `lr_scalers.pkl` - Data scalers

### Evaluation Reports (`reports/`)
- `metrics.csv` - All evaluation metrics
- `predictions_lr.csv` - Test set predictions
- `roc_curve_lr.png` - ROC curve plot
- `pr_curve_lr.png` - Precision-Recall curve plot
- `confusion_matrix_lr.png` - Confusion matrix heatmap
- `calibration_curve_lr.png` - Probability calibration curve
- `feature_importance_lr.png` - Feature importance plot

---

## 🚀 Next Steps

### 1. View Visualization Results
```bash
cd YuchenZhou_Pipeline/training/reports
open roc_curve_lr.png
open confusion_matrix_lr.png
open feature_importance_lr.png
```

### 2. Train More Models

#### Quick Train Random Forest
```bash
cd YuchenZhou_Pipeline/training
python src/train.py --model rf --config config.yaml
```

#### Quick Train XGBoost
```bash
python src/train.py --model xgb --config config.yaml
```

#### Train All Traditional Models (Recommended)
```bash
python src/train.py --model logistic,rf,xgb --config config.yaml
```

#### Train All Models (Including Deep Learning, ~1 hour)
```bash
python src/train.py --model all --config config.yaml
```

### 3. Use Trained Model for Predictions

```bash
cd ../testing
python src/inference.py \
  --model ../training/artifacts/lr.pkl \
  --data ../../cleaned_data.csv \
  --output predictions.csv
```

### 4. Adjust Feature Selection Parameters

Edit `config.yaml`:

```yaml
feature_selection:
  enabled: true
  method: "importance_file"
  top_n: 100  # Increase number of features
  importance_threshold: 0.01  # Lower threshold
```

Then retrain:
```bash
python src/train.py --model logistic --config config.yaml
```

### 5. Model Comparison and Analysis

After training multiple models, view comparison results:
- `reports/metrics.csv` - Contains metrics for all models
- `reports/model_comparison.png` - Model comparison visualization

---

## 💡 Recommendations

### Improving Model Performance
1. **Handle Class Imbalance**:
   - Enable SMOTE in `config.yaml`: `use_smote: true`
   - Adjust class weights: `class_weight: balanced`

2. **Adjust Decision Threshold**:
   - Current default threshold 0.5
   - Can adjust to 0.3-0.4 to improve recall
   - Or adjust to 0.6-0.7 to improve precision

3. **Increase Features**:
   - Try `top_n: 100` to use more features
   - Or set `enabled: false` to use all features for comparison

4. **Try More Powerful Models**:
   - Random Forest - Usually better performance
   - XGBoost - Good at handling imbalanced data
   - Deep learning models - May capture complex patterns

### Model Deployment
1. Use `testing/src/inference.py` for batch predictions
2. Model files are in `artifacts/` directory
3. Can be directly used for predictions on new data

---

## 📝 Technical Details

### Preprocessing Pipeline
1. **Data Loading**: Read MIMIC-IV data from CSV
2. **Feature Selection**: Select top-50 features based on LASSO coefficients
3. **Feature Mapping**: Map One-Hot encoded names back to original column names
4. **Missing Value Handling**: 
   - Categorical features: Fill with "Unknown"
   - Numerical features: Fill with median
5. **Encoding**: 
   - One-Hot encoding: gender, marital_status, insurance, admission_type
   - High cardinality features removed: last_service, language, admission_location, discharge_location
6. **Standardization**: StandardScaler normalization for numerical features

### Model Parameters
- **Solver**: lbfgs
- **Penalty**: L2 regularization
- **Max Iterations**: 1000
- **Class Weight**: Balanced (automatically handles class imbalance)

---

## 🎓 Project Information

- **Course**: CS526 - Fall 2025
- **Project**: 30-day Hospital Readmission Prediction
- **Dataset**: MIMIC-IV
- **Student**: Yuchen Zhou
- **Collaboration**: Based on Xi Chen's LASSO feature selection results

---

Generated Time: 2025-01-XX
Pipeline Version: v1.0
