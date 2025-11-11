#!/bin/bash
# Monitor preprocessing progress

cd "$(dirname "$0")"

echo "Monitoring preprocessing progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================================================"
    echo "MIMIC-IV Preprocessing Monitor - $(date)"
    echo "========================================================================"
    echo ""
    
    echo "Output files:"
    ls -lh ../output/*.parquet ../output/*.pkl 2>/dev/null | awk '{print $9, $5}' | sed 's|../output/||'
    
    echo ""
    echo "Latest log entries (if any):"
    if [ -f "step2_6_log.txt" ]; then
        tail -20 step2_6_log.txt
    else
        echo "No log file found yet"
    fi
    
    echo ""
    echo "========================================================================"
    echo "Refreshing in 30 seconds..."
    sleep 30
done
