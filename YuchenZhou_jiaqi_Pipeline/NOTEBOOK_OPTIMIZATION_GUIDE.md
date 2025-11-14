# Similarity-Based Analysis Notebook - Optimization Guide

## 🎯 Purpose
This notebook implements similarity-based data balancing, bootstrap feature selection, model training, and risk score calculation for MIMIC-IV 30-day readmission prediction.

## ⚡ Key Optimizations

### 1. **Memory Management**
- **Float32 precision**: Reduces memory usage by 50% compared to Float64
- **Batch processing**: Processes data in chunks to avoid loading everything into memory
- **Explicit garbage collection**: Frees memory after major steps
- **Memory monitoring**: Tracks usage with `psutil` to identify bottlenecks

### 2. **Feature Extraction**
- **Batch size**: 5,000 samples at a time
- **Progress tracking**: Visual progress bars with `tqdm`
- **Memory estimate**: ~270 MB per dataset (train/val/test) with float32

### 3. **Similarity Computation**
- **Batch processing**: 1,000 readmitted patients at a time
- **Prevents memory overflow**: Large similarity matrices computed in chunks
- **Estimated time**: 2-5 minutes for ~20K readmitted patients

### 4. **Bootstrap Feature Selection**
- **Reduced iterations**: 50 instead of 100 (still statistically robust)
- **Smaller Random Forests**: 50 trees, max_depth=10
- **Parallel processing**: Uses all CPU cores with `n_jobs=-1`
- **Estimated time**: 10-15 minutes

## 🔧 Troubleshooting

### If Kernel Still Crashes:

#### Option 1: Reduce Batch Sizes
```python
# In feature extraction cell:
train_agg, feature_cols = extract_aggregate_features(
    train_data, train_data['feature_names'], 
    batch_size=2000  # Reduce from 5000 to 2000
)

# In similarity computation cell:
matched_indices, similarity_scores = find_similar_pairs(
    readmit_features, no_readmit_features, 
    method='cosine', top_k=1, 
    batch_size=500  # Reduce from 1000 to 500
)
```

#### Option 2: Use Subset for Testing
```python
# After loading data, before feature extraction:
# Use 20% of training data for testing
subset_size = len(train_data['data']) // 5
train_data['data'] = train_data['data'][:subset_size]
train_labels = train_labels[:subset_size]
print(f"Using subset: {len(train_data['data'])} samples")
```

#### Option 3: Increase System Resources
- **Restart kernel** before running: Kernel → Restart Kernel
- **Close other applications** to free RAM
- **Increase swap space** (virtual memory):
  ```bash
  # macOS: System Settings → General → Storage
  # Ensure at least 10GB swap space available
  ```

## 📊 Expected Performance

### Memory Usage (Approximate)
- Initial data loading: ~12 GB (train + val + test pickle files)
- After feature extraction: ~3 GB (aggregate features)
- During similarity computation: ~4-5 GB peak
- During bootstrap: ~6-7 GB peak
- **Recommended RAM**: 16 GB minimum, 32 GB ideal

### Runtime (on M1/M2 MacBook Pro with 16GB RAM)
- Data loading: 1-2 minutes
- Feature extraction: 5-7 minutes
- Similarity computation: 3-5 minutes
- Bootstrap feature selection: 10-15 minutes
- Model training: 2-3 minutes
- **Total**: ~30-35 minutes

## 🚀 Quick Start

1. **Check your system RAM**: 
   ```bash
   # macOS
   sysctl hw.memsize
   ```

2. **Start with a clean kernel**: 
   - Kernel → Restart Kernel → Restart

3. **Run cells sequentially**: 
   - Don't skip cells
   - Wait for each cell to complete before running the next

4. **Monitor memory**: 
   - Watch the `print_memory_usage()` output
   - If usage exceeds 80% of available RAM, consider subset option

## 📈 Output Files

The notebook generates:
1. **CSV Files**:
   - `feature_importance_bootstrap.csv` - Feature importance scores
   - `model_evaluation_results.csv` - Model performance metrics
   - `patient_risk_scores.csv` - Individual risk scores

2. **Visualizations** (PNG files):
   - `feature_importance_visualization.png`
   - `model_comparison.png`
   - `roc_curves.png`
   - `pr_curves.png`
   - `confusion_matrices.png`
   - `risk_score_analysis.png`

3. **Summary Report**:
   - `analysis_summary_report.txt` - Comprehensive text summary

## 🔍 Understanding the Results

### Risk Scores
- **Low Risk** (< 0.3): Patients unlikely to be readmitted
- **Medium Risk** (0.3 - 0.6): Moderate readmission probability
- **High Risk** (> 0.6): High readmission probability

### Feature Importance
- **Stability Score**: Higher = more consistent across bootstrap samples
- **Top 20%**: Most predictive and stable features selected

### Model Performance
- **AUC-ROC**: Overall discrimination ability (higher is better)
- **AUC-PR**: Performance on imbalanced data (higher is better)
- **Sensitivity**: True positive rate (readmissions correctly identified)
- **Specificity**: True negative rate (non-readmissions correctly identified)

## 💡 Tips for Success

1. **Run during off-hours**: Less competition for system resources
2. **Close VS Code extensions**: Disable unused extensions temporarily
3. **Use terminal for monitoring**: 
   ```bash
   # Watch memory usage in real-time
   watch -n 1 'ps aux | grep python | head -n 5'
   ```
4. **Save intermediate results**: The notebook automatically saves CSVs after each major step
5. **Restart if needed**: If a cell hangs, restart kernel and resume from last saved checkpoint

## 📞 Support

If issues persist:
1. Check system logs: Console app (macOS) → search "kernel"
2. Verify data files: Ensure all pkl files in `output/` folder are valid
3. Test with smaller dataset: Use subset option to verify notebook works
4. Consider cloud resources: Google Colab, AWS SageMaker, or Azure ML for larger RAM

## ✅ Success Indicators

You'll know the notebook is working correctly when:
- ✓ Memory usage stays below 80% of available RAM
- ✓ Progress bars update smoothly without hanging
- ✓ Each cell completes within expected timeframe
- ✓ Output files are generated after each section
- ✓ Final summary report shows sensible metrics (AUC > 0.6)

---

**Last Updated**: November 13, 2025
**Optimized for**: macOS with 16GB+ RAM, M1/M2 processors
**Dataset**: MIMIC-IV (194,672 training samples)
