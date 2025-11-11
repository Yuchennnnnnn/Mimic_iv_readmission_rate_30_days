#!/bin/bash
# Run preprocessing steps 2-6 sequentially

cd "$(dirname "$0")"

echo "========================================================================"
echo "Running MIMIC-IV Preprocessing Steps 2-6"
echo "========================================================================"
echo ""

# Check if Step 1 outputs exist
if [ ! -f "../output/cohort.parquet" ]; then
    echo "❌ ERROR: cohort.parquet not found!"
    echo "   Please run Step 1 first."
    exit 1
fi

if [ ! -f "../output/chartevents_raw.parquet" ]; then
    echo "❌ ERROR: chartevents_raw.parquet not found!"
    echo "   Please run Step 1 first."
    exit 1
fi

if [ ! -f "../output/labevents_raw.parquet" ]; then
    echo "❌ ERROR: labevents_raw.parquet not found!"
    echo "   Please run Step 1 first."
    exit 1
fi

echo "✓ Step 1 outputs verified"
echo ""

# Step 2: Clean units
echo "========================================================================"
echo "Step 2: Cleaning and standardizing units"
echo "========================================================================"
python scripts/step2_clean_units.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Step 2 failed"
    exit 1
fi
echo ""

# Step 3: Create time series
echo "========================================================================"
echo "Step 3: Creating time series (48-hour bins)"
echo "========================================================================"
python scripts/step3_create_timeseries.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Step 3 failed"
    exit 1
fi
echo ""

# Step 4: Compute features
echo "========================================================================"
echo "Step 4: Computing features (masks, deltas, imputation)"
echo "========================================================================"
python scripts/step4_compute_features.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Step 4 failed"
    exit 1
fi
echo ""

# Step 5: Temporal split
echo "========================================================================"
echo "Step 5: Creating temporal train/val/test split"
echo "========================================================================"
python scripts/step5_temporal_split.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Step 5 failed"
    exit 1
fi
echo ""

# Step 6: Save output
echo "========================================================================"
echo "Step 6: Saving final outputs"
echo "========================================================================"
python scripts/step6_save_output.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Step 6 failed"
    exit 1
fi
echo ""

echo "========================================================================"
echo "✅ ALL STEPS COMPLETE!"
echo "========================================================================"
echo ""
echo "Output files in: ../output/"
echo "  - train_data.pkl"
echo "  - val_data.pkl"
echo "  - test_data.pkl"
echo "  - train_index.parquet"
echo "  - val_index.parquet"
echo "  - test_index.parquet"
echo ""
echo "You can now use these files for model training!"
