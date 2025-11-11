#!/bin/bash
# Real-time monitoring script for preprocessing pipeline

cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/YuchenZhou_jiaqi_Pipeline/preprocessing

echo "========================================================================"
echo "MIMIC-IV Preprocessing Pipeline Monitor"
echo "Started at: $(date)"
echo "========================================================================"
echo ""

# Check if process is running
PID=$(pgrep -f "step3_create_timeseries.py|step4_compute_features.py|step5_temporal_split.py|step6_save_output.py")
if [ -n "$PID" ]; then
    echo "✓ Pipeline is running (PID: $PID)"
else
    echo "⚠ No pipeline process detected"
fi

echo ""
echo "========================================================================"
echo "Latest Log Output (last 30 lines):"
echo "========================================================================"
if [ -f "step3_6_full.log" ]; then
    tail -30 step3_6_full.log
else
    echo "Log file not found yet..."
fi

echo ""
echo "========================================================================"
echo "Output Files:"
echo "========================================================================"
ls -lh ../output/*.parquet ../output/*.pkl 2>/dev/null | awk '{printf "%-40s %10s %s %s %s\n", $9, $5, $6, $7, $8}' | sed 's|../output/||'

echo ""
echo "========================================================================"
echo "Disk Usage:"
echo "========================================================================"
du -sh ../output

echo ""
echo "========================================================================"
echo "To continue monitoring, run:"
echo "  tail -f step3_6_full.log"
echo "========================================================================"
