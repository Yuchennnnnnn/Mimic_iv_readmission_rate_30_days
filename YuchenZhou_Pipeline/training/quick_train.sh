#!/bin/bash
# Quick Training Script - Yuchen Zhou's Readmission Prediction Pipeline

echo "=========================================="
echo "  30-Day Readmission Prediction"
echo "  Yuchen Zhou's Pipeline"
echo "=========================================="
echo ""

# Set virtual environment Python path
VENV_PYTHON="/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/.venv/bin/python"

# Check virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Please create virtual environment first: python -m venv .venv"
    exit 1
fi

# Enter training directory
cd "$(dirname "$0")"

echo "Select training mode:"
echo ""
echo "  1. Quick test (Logistic Regression only, ~2 minutes)"
echo "  2. Traditional ML models (LR + RF + XGBoost, ~15 minutes)"
echo "  3. All models (including LSTM and Transformer, ~1 hour)"
echo "  4. Custom"
echo "  5. Exit"
echo ""
read -p "Please select [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting quick test..."
        $VENV_PYTHON src/train.py --model logistic --config config.yaml
        ;;
    2)
        echo ""
        echo "🚀 Training traditional ML models..."
        $VENV_PYTHON src/train.py --model logistic --config config.yaml
        $VENV_PYTHON src/train.py --model rf --config config.yaml
        $VENV_PYTHON src/train.py --model xgb --config config.yaml
        ;;
    3)
        echo ""
        echo "🚀 Training all models (this will take a while)..."
        $VENV_PYTHON src/train.py --model all --config config.yaml
        ;;
    4)
        echo ""
        echo "Available models: logistic, rf, xgb, lstm, transformer, all"
        read -p "Enter model name: " model_name
        echo ""
        echo "🚀 Training $model_name ..."
        $VENV_PYTHON src/train.py --model "$model_name" --config config.yaml
        ;;
    5)
        echo "Exit"
        exit 0
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac

# Check training results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "📊 View results:"
    echo "   Metrics: reports/metrics.csv"
    echo "   Visualizations: reports/*.png"
    echo "   Models: artifacts/*.pkl"
    echo ""
    echo "📖 Detailed documentation: ../TRAINING_RESULTS.md"
else
    echo ""
    echo "=========================================="
    echo "  ❌ Training failed"
    echo "=========================================="
    exit 1
fi
