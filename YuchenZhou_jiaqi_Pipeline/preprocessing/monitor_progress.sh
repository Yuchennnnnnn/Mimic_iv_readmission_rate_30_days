#!/bin/bash
# Monitor preprocessing progress

OUTPUT_DIR="../output"

echo "========================================================================"
echo "Preprocessing Progress Monitor"
echo "========================================================================"
echo ""

# Check which files exist
echo "Output files:"
for file in cohort.parquet chartevents_raw.parquet labevents_raw.parquet prescriptions_raw.parquet \
            chartevents_clean.parquet labevents_clean.parquet prescriptions_clean.parquet \
            timeseries_binned.pkl timeseries_features.pkl \
            train_temp.pkl val_temp.pkl test_temp.pkl \
            train_data.pkl val_data.pkl test_data.pkl; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    fi
done

echo ""
echo "Disk space in output directory:"
du -sh "$OUTPUT_DIR"

echo ""
echo "Most recent log entries in step2_6_log.txt:"
if [ -f "step2_6_log.txt" ]; then
    tail -20 step2_6_log.txt
else
    echo "  No log file found"
fi
