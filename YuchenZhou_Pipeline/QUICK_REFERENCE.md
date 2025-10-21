# 🚀 QUICK REFERENCE CARD

## Installation & Setup (One Time)
```bash
cd training
pip install -r requirements.txt
```

## Quick Test (2 minutes)
```bash
cd training
python quick_start.py
```

## Train Models

### Train All Models
```bash
cd training
python src/train.py --model all --config config.yaml
```

### Train Individual Models
```bash
# Fastest (1 min)
python src/train.py --model logistic

# Best performance (5 min)
python src/train.py --model xgb

# With custom epochs
python src/train.py --model lstm --epochs 30 --batch-size 64
```

## View Results
```bash
# Metrics table
cat reports/metrics.csv

# Open plots (macOS)
open reports/model_comparison.png
open reports/roc_curve_xgb.png

# Open plots (Linux)
xdg-open reports/model_comparison.png
```

## Make Predictions
```bash
cd testing
python src/inference.py \
  --model-path ../training/artifacts/xgb.pkl \
  --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \
  --input new_data.csv \
  --output predictions.csv \
  --model-type sklearn
```

## Configuration

### Update Data Path
```yaml
# In config.yaml
data:
  input_path: "../cleaned_data.csv"  # Your data here
```

### Adjust Hyperparameters
```yaml
# For imbalanced data
hyperparameters:
  xgb:
    scale_pos_weight: 5.0  # Higher for more imbalance

# For faster training
  lstm:
    num_epochs: 20      # Reduce from 50
    batch_size: 128     # Increase from 64
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
hyperparameters:
  lstm:
    batch_size: 32
```

### Import Errors
```bash
# Run from training directory
cd training
python src/train.py ...
```

### Slow Training
```bash
# Use CPU only
export CUDA_VISIBLE_DEVICES=""

# Or select GPU
export CUDA_VISIBLE_DEVICES=0
```

## File Locations

```
training/
├── src/train.py           ← Main training script
├── config.yaml            ← Configuration
├── artifacts/             ← Saved models
├── reports/               ← Results & plots
└── README.md              ← Full documentation

testing/
├── src/inference.py       ← Prediction script
└── artifacts/             ← Copy models here
```

## Output Files

### After Training
```
artifacts/lr.pkl                    # Models
reports/metrics.csv                 # All metrics
reports/model_comparison.png        # Comparison chart
reports/roc_curve_*.png            # ROC curves
reports/predictions_*.csv          # Predictions
```

## Key Commands Summary

| Action | Command |
|--------|---------|
| Quick test | `python quick_start.py` |
| Train best model | `python src/train.py --model xgb` |
| Train all | `python src/train.py --model all` |
| View metrics | `cat reports/metrics.csv` |
| Make predictions | `python ../testing/src/inference.py ...` |
| Run tests | `pytest tests/ -v` |

## Model Comparison Quick Reference

| Model | Speed | Performance | Use When |
|-------|-------|-------------|----------|
| **Logistic** | ⚡⚡⚡ | ⭐⭐ | Need interpretability |
| **Random Forest** | ⚡⚡ | ⭐⭐⭐ | Want feature importance |
| **XGBoost** | ⚡⚡ | ⭐⭐⭐⭐ | **Best overall** |
| **LSTM** | ⚡ | ⭐⭐⭐ | Sequential patterns |
| **Transformer** | ⚡ | ⭐⭐⭐ | Feature interactions |

## Typical Performance (MIMIC-IV)

```
Model            Time     ROC-AUC  F1-Score
──────────────────────────────────────────
Logistic         1 min    0.67     0.38
Random Forest    5 min    0.71     0.41
XGBoost          5 min    0.73     0.43  ⭐
LSTM            30 min    0.71     0.42
Transformer     40 min    0.72     0.42
```

## Support

📖 Full docs: `PIPELINE_README.md`  
📖 Training guide: `training/README.md`  
📖 Summary: `IMPLEMENTATION_SUMMARY.md`  

---
*Keep this file handy for quick reference!*
