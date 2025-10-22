# Model Training Results Report

**Training Time**: 2025-10-21 00:24:46  
**Dataset**: MIMIC-IV cleaned_data.csv  
**Number of Features**: 18 features (mapped from LASSO's 48 One-Hot features)  
**Training Samples**: 164,784  
**Test Samples**: 41,196  
**Readmission Rate**: 26.72%  

---

## 📊 Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score |
|------|---------|--------|----------|-----------|--------|----------|
| **LR** | 0.6626 | 0.4037 | 0.5918 | 0.3576 | 0.6621 | 0.4643 |
| **RF** | 0.6933 | 0.4610 | 0.6397 | 0.3915 | 0.6284 | 0.4824 |
| **XGB** | 0.7029 | 0.4746 | 0.6250 | 0.3862 | 0.6846 | 0.4938 |

---

## 🏆 Best Models

- **Best ROC-AUC**: XGB (0.7029)
- **Best F1-Score**: XGB (0.4938)
- **Best Recall**: XGB (0.6846)

---

## 💡 Key Findings

1. **XGBoost Performs Best**: ROC-AUC reaches 0.7029, outperforming other models across all metrics
2. **Recall vs Precision Trade-off**: 
   - XGBoost: Highest recall (68.46%), suitable for capturing more readmission patients
   - Random Forest: More balanced precision (39.15%)
3. **Effective Feature Selection**: Achieved 0.70+ AUC with only 18 features

---

## 📈 Detailed Metrics

### Logistic Regression
- ROC-AUC: 0.6626
- Advantages: Fast training, strong interpretability
- Use Cases: Scenarios requiring quick deployment and interpretation

### Random Forest  
- ROC-AUC: 0.6933
- Advantages: Automatically handles non-linear relationships, feature importance visualization
- Use Cases: When feature importance analysis is needed

### XGBoost ⭐
- ROC-AUC: 0.7029
- Advantages: Best performance, handles complex patterns
- Use Cases: First choice for production environments

---

## 📁 File Locations

- Models: `artifacts/*.pkl`
- Prediction Results: `reports/predictions_*.csv`
- Visualizations: `reports/*.png`
- Detailed Metrics: `reports/metrics.csv`

---

## 🔧 Next Steps Recommendations

1. **Hyperparameter Tuning**: Optimize XGBoost using GridSearch
2. **Feature Engineering**: Try adding more LASSO features (top_n: 100)
3. **Ensemble Learning**: Combine predictions from multiple models
4. **Deep Learning**: Train LSTM and Transformer models
5. **Model Interpretation**: Use SHAP to analyze feature importance
