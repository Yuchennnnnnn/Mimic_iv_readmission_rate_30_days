# Similarity-Based Analysis for MIMIC-IV Readmission Prediction

## 📋 Overview

This notebook implements a comprehensive analysis pipeline for 30-day hospital readmission prediction using similarity-based data balancing, bootstrap feature selection, and ensemble modeling.

## 🎯 What It Does

1. **Similarity-Based Data Balancing**: Matches each readmitted patient with the most similar non-readmitted patient using cosine similarity
2. **Bootstrap Feature Selection**: Identifies the most stable and important features across 50 bootstrap iterations
3. **Multi-Model Training**: Trains Logistic Regression, Random Forest, and Gradient Boosting models
4. **Risk Score Calculation**: Generates patient-level risk scores with Low/Medium/High stratification
5. **Comprehensive Evaluation**: Produces AUC-ROC, AUC-PR, confusion matrices, and calibration plots

## 🚀 Quick Start

### Option 1: Run Pre-flight Check First (Recommended)

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/YuchenZhou_jiaqi_Pipeline

# Test if your system can handle the notebook
python3 test_notebook_requirements.py
```

This will check:
- ✓ Available memory (needs 12GB+ free)
- ✓ Data files exist and are valid
- ✓ Required packages are installed
- ✓ Can load data successfully

### Option 2: Run Directly in VS Code

1. Open `similarity_based_analysis.ipynb` in VS Code
2. Select your Python kernel (conda: base or miniconda3)
3. Run all cells sequentially (Shift + Enter)
4. Wait ~30-35 minutes for completion

### Option 3: Run in Jupyter

```bash
jupyter notebook similarity_based_analysis.ipynb
```

## ⚙️ System Requirements

### Minimum Requirements
- **RAM**: 16 GB (8 GB+ available)
- **Storage**: 50 GB free space
- **CPU**: 4+ cores recommended
- **Time**: ~30-35 minutes runtime

### Optimal Requirements
- **RAM**: 32 GB (12 GB+ available)
- **Storage**: 100 GB free space
- **CPU**: 8+ cores
- **Time**: ~20-25 minutes runtime

## 📦 Dependencies

Auto-installed by notebook:
- `psutil` - Memory monitoring
- `tqdm` - Progress bars

Pre-installed (should already have):
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Scientific computing

Install missing packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy tqdm psutil
```

## 📊 Expected Outputs

### CSV Files (in current directory)
1. `feature_importance_bootstrap.csv` - 343 features ranked by stability
2. `model_evaluation_results.csv` - Performance metrics for all models
3. `patient_risk_scores.csv` - Risk scores for 44,790 test patients

### Visualizations (PNG files)
1. `feature_importance_visualization.png` - Top 30 features with error bars
2. `model_comparison.png` - 6-panel comparison of all metrics
3. `roc_curves.png` - ROC curves for all 3 models
4. `pr_curves.png` - Precision-Recall curves
5. `confusion_matrices.png` - 3 confusion matrices
6. `risk_score_analysis.png` - Risk distribution and calibration

### Summary Report
- `analysis_summary_report.txt` - Complete text summary of all results

## 🔍 Understanding the Results

### Risk Stratification
- **Low Risk (< 0.3)**: Patients safe for discharge
- **Medium Risk (0.3-0.6)**: May need follow-up monitoring
- **High Risk (> 0.6)**: Require intervention programs

### Feature Importance
Check `feature_importance_bootstrap.csv`:
- **Stability Score**: Higher = more reliable predictor
- **Mean Importance**: Average importance across bootstrap samples
- **Top features**: Typically vital signs (heart_rate, sbp) and lab values (creatinine, bun)

### Model Performance
Expected ranges (test set):
- **AUC-ROC**: 0.65-0.75 (good discrimination)
- **AUC-PR**: 0.30-0.45 (better than baseline ~0.15)
- **Sensitivity**: 0.60-0.75 (catch most readmissions)
- **Specificity**: 0.60-0.75 (minimize false alarms)

## ⚠️ Troubleshooting

### Kernel Crashes or Hangs

**Symptom**: Kernel dies during execution

**Solutions**:
1. **Restart kernel** and try again (often works on second attempt)
2. **Close other applications** to free memory
3. **Reduce batch sizes**:
   - Change `batch_size=5000` to `2000` in feature extraction cell
   - Change `batch_size=1000` to `500` in similarity computation cell
4. **Use subset for testing**:
   ```python
   # After loading train_data, add:
   train_data['data'] = train_data['data'][:50000]  # Use first 50K samples
   ```

### Out of Memory Errors

**Symptom**: "MemoryError" or system becomes very slow

**Solutions**:
1. Monitor memory with Activity Monitor (macOS) or Task Manager (Windows)
2. Ensure 12GB+ RAM is free before starting
3. Run during off-hours when system is idle
4. See "Reduce batch sizes" above

### Slow Execution

**Symptom**: Cells take much longer than expected

**Reasons**:
- Other processes competing for CPU/RAM
- Using older hardware (< 2018)
- Running on battery power (macOS throttles CPU)

**Solutions**:
- Close unnecessary applications
- Connect to power (important on laptops!)
- Use `top` or Activity Monitor to identify competing processes

### Missing Dependencies

**Symptom**: "ModuleNotFoundError: No module named 'X'"

**Solution**:
```bash
pip install X  # Replace X with missing package name
```

Or install all at once:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy tqdm psutil
```

## 📈 Performance Benchmarks

Tested on M1 MacBook Pro (16GB RAM):

| Stage | Time | Peak Memory |
|-------|------|-------------|
| Data loading | 1-2 min | 12 GB |
| Feature extraction | 5-7 min | 3 GB |
| Similarity computation | 3-5 min | 5 GB |
| Bootstrap (50 iter) | 10-15 min | 7 GB |
| Model training | 2-3 min | 4 GB |
| **Total** | **~30 min** | **~12 GB** |

Intel Macs or older hardware may be 1.5-2x slower.

## 🎓 Methodology

### Why Similarity-Based Balancing?
Traditional methods (SMOTE, undersampling) either create synthetic data or lose information. Similarity-based matching:
- ✓ Preserves real patient data
- ✓ Ensures matched controls are clinically similar
- ✓ Reduces selection bias
- ✓ Creates 1:1 balanced dataset naturally

### Why Bootstrap Feature Selection?
Standard feature importance can be unstable. Bootstrap approach:
- ✓ Tests feature importance across 50 different samples
- ✓ Identifies features that are consistently important
- ✓ Calculates stability score (mean/std)
- ✓ Selects top 20% most stable features

### Why Ensemble Models?
Different algorithms capture different patterns:
- **Logistic Regression**: Linear relationships, interpretable
- **Random Forest**: Non-linear relationships, robust to outliers
- **Gradient Boosting**: Sequential learning, high accuracy
- **Ensemble Average**: Combines strengths, reduces individual model weaknesses

## 📖 Further Reading

See related documentation:
- `NOTEBOOK_OPTIMIZATION_GUIDE.md` - Detailed optimization explanations
- `../preprocessing/README.md` - How the input data was generated
- `../training/README.md` - Alternative deep learning approaches

## 💬 Interpretation Tips

### For Clinicians
- Focus on **Risk Stratification** results (Section 7, cell 28)
- Check **Feature Importance** to understand clinical drivers (Section 4, cell 16)
- Review **Calibration Plot** to assess if probabilities are reliable (Section 7, cell 27)

### For Data Scientists
- Examine **AUC-ROC vs AUC-PR** trade-offs (Section 6, cells 21-23)
- Compare **model complexity vs performance** (simpler may be better for deployment)
- Check **Confusion Matrices** to tune thresholds for your use case (Section 6, cell 24)

### For Researchers
- Review **Bootstrap stability scores** for reproducibility (Section 4, cell 16)
- Check **similarity distribution** to assess matching quality (Section 3, cell 11)
- Examine **feature importance** to generate hypotheses for validation (Section 4, cell 17)

## ✅ Success Criteria

Your notebook run was successful if:
1. ✓ All cells execute without errors
2. ✓ 9 output files are generated (3 CSV + 6 PNG)
3. ✓ Test AUC-ROC > 0.60 (better than random)
4. ✓ Risk stratification shows clear separation (Low < Medium < High readmission rates)
5. ✓ Summary report looks reasonable (no NaN values, sensible numbers)

## 🐛 Known Issues

1. **Widget Error** (CDN fetch failed): Harmless warning, doesn't affect execution
2. **tqdm progress bars**: May not display in some environments (functionality still works)
3. **Memory usage varies**: Depends on system load and background processes

## 📞 Support

If you encounter issues:
1. Run `test_notebook_requirements.py` to diagnose problems
2. Check `NOTEBOOK_OPTIMIZATION_GUIDE.md` for detailed troubleshooting
3. Review VS Code Python extension logs: Output → Python
4. Try subset approach if memory is limited

---

**Author**: AI Assistant  
**Date**: November 13, 2025  
**Version**: 1.0 (Memory-Optimized)  
**Dataset**: MIMIC-IV v2.0 (194,672 train, 44,790 test samples)
