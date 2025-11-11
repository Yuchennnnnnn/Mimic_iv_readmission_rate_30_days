#!/bin/bash
# Run all preprocessing steps sequentially

echo "========================================================================"
echo "MIMIC-IV Temporal Preprocessing Pipeline (Pure Python)"
echo "========================================================================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

# Create output directory
mkdir -p output

echo ""
echo "Step 1: Loading MIMIC-IV data and creating cohort..."
echo "------------------------------------------------------------------------"
python scripts/step1_load_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 failed"
    exit 1
fi

echo ""
echo "Step 2: Cleaning and standardizing units..."
echo "------------------------------------------------------------------------"
python scripts/step2_clean_units.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi

echo ""
echo "Step 3: Creating fixed-length time series..."
echo "------------------------------------------------------------------------"
python scripts/step3_create_timeseries.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed"
    exit 1
fi

echo ""
echo "Step 4: Computing masks, deltas, and imputation..."
echo "------------------------------------------------------------------------"
python scripts/step4_compute_features.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 failed"
    exit 1
fi

echo ""
echo "Step 5: Temporal split by anchor_year_group..."
echo "------------------------------------------------------------------------"
python scripts/step5_temporal_split.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 5 failed"
    exit 1
fi

echo ""
echo "Step 6: Saving final output..."
echo "------------------------------------------------------------------------"
python scripts/step6_save_output.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 6 failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ All preprocessing steps completed successfully!"
echo "========================================================================"
echo ""
echo "Output files saved to: output/"
echo ""
echo "Generated files:"
echo "  - train_data.pkl, train_index.parquet"
echo "  - val_data.pkl, val_index.parquet"
echo "  - test_data.pkl, test_index.parquet"
echo ""
echo "Next steps:"
echo "  1. Load data in your training script"
echo "  2. Create PyTorch Dataset/DataLoader"
echo "  3. Train your model!"
echo ""
