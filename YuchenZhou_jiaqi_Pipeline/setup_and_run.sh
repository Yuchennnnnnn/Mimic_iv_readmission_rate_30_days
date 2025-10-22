#!/bin/bash
# Quick setup script for the readmission prediction pipeline

echo "=========================================="
echo "  30-Day Readmission Prediction Pipeline"
echo "  Setup & Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Navigate to training directory
cd training

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎯 Running quick start test..."
echo ""

# Run quick start
python3 quick_start.py

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "📁 Check these directories:"
echo "   - artifacts/     (saved models)"
echo "   - reports/       (metrics and plots)"
echo "   - data/          (test data)"
echo ""
echo "🚀 Next steps:"
echo "   1. Train on real data:"
echo "      python src/train.py --model all --config config.yaml"
echo ""
echo "   2. View results:"
echo "      cat reports/metrics.csv"
echo ""
echo "   3. Try other models:"
echo "      python src/train.py --model xgb --config config.yaml"
echo ""
echo "📖 For detailed documentation, see:"
echo "   - README.md (in training/ directory)"
echo "   - PIPELINE_README.md (in project root)"
echo ""
echo "=========================================="
