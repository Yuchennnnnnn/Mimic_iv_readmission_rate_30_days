#!/bin/bash
# Quick inference test - using trained XGBoost model

echo "=========================================="
echo "  Model Inference Testing"
echo "  Using XGBoost model to predict 30-day readmission risk"
echo "=========================================="
echo ""

# Set virtual environment Python path
VENV_PYTHON="/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/.venv/bin/python"

# Check virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found"
    exit 1
fi

# Enter testing directory
cd "$(dirname "$0")"

# Check model files
if [ ! -f "../training/artifacts/xgb.pkl" ]; then
    echo "❌ Error: XGBoost model not found"
    echo "   Please train the model first: cd ../training && python src/train.py --model xgb"
    exit 1
fi

echo "✓ Found XGBoost model"
echo "✓ Found data file"
echo ""

echo "Starting inference..."
echo "=========================================="
echo ""

# Run inference
$VENV_PYTHON src/inference.py \
    --model-path ../training/artifacts/xgb.pkl \
    --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \
    --input ../../cleaned_data.csv \
    --output reports/predictions_xgb.csv \
    --model-type sklearn

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ Inference completed successfully!"
    echo "=========================================="
    echo ""
    echo "📊 View results:"
    echo "   Prediction file: reports/predictions_xgb.csv"
    echo ""
    
    # Display first few prediction results
    if [ -f "reports/predictions_xgb.csv" ]; then
        echo "Prediction results example (first 10 rows):"
        echo "----------------------------------------"
        head -n 11 reports/predictions_xgb.csv | column -t -s,
        echo "----------------------------------------"
        echo ""
        
        # Statistics
        echo "Prediction statistics:"
        $VENV_PYTHON -c "
import pandas as pd
df = pd.read_csv('reports/predictions_xgb.csv')
print(f'  Total samples: {len(df):,}')
print(f'  Predicted readmission: {df[\"predicted_label\"].sum():,} ({df[\"predicted_label\"].mean()*100:.1f}%)')
print(f'  Average prediction probability: {df[\"predicted_probability\"].mean():.3f}')
print(f'  Probability range: [{df[\"predicted_probability\"].min():.3f}, {df[\"predicted_probability\"].max():.3f}]')
"
    fi
else
    echo ""
    echo "=========================================="
    echo "  ❌ Inference failed"
    echo "=========================================="
    exit 1
fi
