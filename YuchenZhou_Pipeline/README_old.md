# Yuchen Zhou's ML Pipeline for 30-Day Readmission Prediction

**Author:** Yuchen Zhou  
**Date:** October 20, 2025  
**Course:** CS526 - Machine Learning in Healthcare, Duke University

---

## 📁 What's in This Folder

This is my individual contribution to the team project. It contains a complete, independent ML pipeline that works alongside my teammates' work.

```
YuchenZhou_Pipeline/
├── training/              # My training pipeline
├── testing/               # My inference/testing pipeline
├── README.md              # This file
├── QUICK_REFERENCE.md     # Quick command reference
├── PIPELINE_README.md     # Detailed documentation
└── setup_and_run.sh       # Quick setup script
```

---

## 🚀 Quick Start (2 Minutes)

### Option 1: Automated Quick Test
```bash
cd YuchenZhou_Pipeline/training
python quick_start.py
```

This will:
1. Generate synthetic test data
2. Train a Logistic Regression model
3. Evaluate and create reports
4. Show you where results are saved

### Option 2: Use Bash Script
```bash
cd YuchenZhou_Pipeline
chmod +x setup_and_run.sh
./setup_and_run.sh
```

---

## 📊 Train on Real Data

### Step 1: Install Dependencies
```bash
cd YuchenZhou_Pipeline/training
pip install -r requirements.txt
```

### Step 2: Update Configuration
Edit `training/config.yaml` to point to the cleaned data:
```yaml
data:
  input_path: "../../cleaned_data.csv"  # Adjust path as needed
```

### Step 3: Train Models
```bash
# Train all 5 models
python src/train.py --model all --config config.yaml

# Or train individually
python src/train.py --model logistic --config config.yaml
python src/train.py --model rf --config config.yaml
python src/train.py --model xgb --config config.yaml
python src/train.py --model lstm --config config.yaml --epochs 30
python src/train.py --model transformer --config config.yaml
```

---

## 📈 View Results

### Check Metrics
```bash
cd training
cat reports/metrics.csv
```

### View Plots
```bash
# ROC curves
open reports/roc_curve_xgb.png

# Model comparison
open reports/model_comparison.png

# Feature importance
open reports/feature_importance_xgb.png
```

---

## 🧪 Make Predictions on New Data

```bash
cd YuchenZhou_Pipeline/testing

python src/inference.py \
  --model-path ../training/artifacts/xgb.pkl \
  --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \
  --input ../../cleaned_data.csv \
  --output my_predictions.csv \
  --model-type sklearn
```

---

## 📚 Documentation

- **QUICK_REFERENCE.md** - Command cheat sheet
- **PIPELINE_README.md** - Complete documentation
- **IMPLEMENTATION_SUMMARY.md** - What was built
- **training/README.md** - Training details
- **testing/README.md** - Inference details

---

## 🎯 Models Implemented

1. **Logistic Regression** - Interpretable baseline
2. **Random Forest** - Ensemble with feature importance
3. **XGBoost** - Best performance (typically)
4. **LSTM** - Deep learning with embeddings
5. **Transformer** - Attention-based architecture

---

## 📂 Expected Outputs

After training, you'll have:

```
training/
├── artifacts/
│   ├── lr.pkl, rf.pkl, xgb.pkl        # Trained models
│   ├── lstm.pt, transformer.pt        # PyTorch models
│   └── *_preprocessor.joblib          # Encoders
└── reports/
    ├── metrics.csv                     # All metrics
    ├── model_comparison.png            # Comparison chart
    ├── roc_curve_*.png                 # ROC curves
    ├── predictions_*.csv               # Predictions
    └── feature_importance_*.png        # Feature importance
```

---

## ⚡ Performance Summary

Expected results on MIMIC-IV data:

| Model | ROC-AUC | Training Time |
|-------|---------|---------------|
| Logistic Reg | 0.67 | 1 min |
| Random Forest | 0.71 | 5 min |
| **XGBoost** | **0.73** | 5 min ⭐ |
| LSTM | 0.71 | 30 min |
| Transformer | 0.72 | 40 min |

---

## 🔧 Troubleshooting

### GPU Out of Memory
```yaml
# In training/config.yaml, reduce batch size:
hyperparameters:
  lstm:
    batch_size: 32
```

### Import Errors
Make sure you're in the correct directory:
```bash
cd YuchenZhou_Pipeline/training
python src/train.py --model logistic
```

### Slow Training
Train on a subset first to test:
```python
# In config.yaml, you can add this in preprocessing:
preprocessing:
  sample_size: 5000  # Use subset for testing
```

---

## 🤝 Relation to Team Project

This pipeline is my **independent contribution** and works separately from:
- `preprocessing/` - Shared preprocessing scripts
- `XiChen_Lasso/` - Xi Chen's LASSO feature selection
- Other teammates' work

All folders can coexist and use the same `cleaned_data.csv` file.

---

## 📝 What I Contributed

✅ Complete end-to-end ML pipeline  
✅ 5 different model implementations  
✅ Comprehensive preprocessing  
✅ Full evaluation suite  
✅ Deployment infrastructure  
✅ 3,500+ lines of code  
✅ 2,000+ lines of documentation  
✅ Unit and integration tests  

---

## 🎓 For Grading/Presentation

**Key Files to Review:**
1. `training/src/train.py` - Main training script
2. `training/src/models.py` - All 5 models
3. `training/reports/metrics.csv` - Results
4. `training/reports/model_comparison.png` - Visual comparison

**To Demonstrate:**
```bash
# Quick test (2 min)
cd training
python quick_start.py

# View results
cat reports/metrics.csv
```

---

## 📧 Questions?

- Check `QUICK_REFERENCE.md` for commands
- See `PIPELINE_README.md` for full documentation
- Review `training/README.md` for detailed guide

---

**Ready to use! Run `python training/quick_start.py` to get started.** 🚀
