#!/bin/bash
# Monitor Step 1 preprocessing progress

echo "======================================================================"
echo "Monitoring MIMIC-IV Step 1 Preprocessing"
echo "======================================================================"
echo ""

# Check if process is running
PID=$(ps aux | grep step1_load_data_optimized.py | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ Process not running!"
    echo ""
    echo "Check the log file:"
    echo "  tail -100 step1_optimized.log"
    exit 1
else
    echo "✅ Process running (PID: $PID)"
    echo ""
fi

# Show process stats
echo "Process Stats:"
ps aux | grep step1_load_data_optimized.py | grep -v grep | awk '{printf "  CPU: %s%%  Memory: %s%%  Time: %s\n", $3, $4, $10}'
echo ""

# Show output directory size
echo "Output Directory:"
ls -lh output/ 2>/dev/null || echo "  (empty or doesn't exist yet)"
echo ""

# Show last 20 lines of log
echo "Latest Log Output (last 20 lines):"
echo "----------------------------------------------------------------------"
tail -20 step1_optimized.log 2>/dev/null || echo "  (log file empty or doesn't exist yet)"
echo "----------------------------------------------------------------------"
echo ""

echo "To watch in real-time:"
echo "  tail -f step1_optimized.log"
echo ""
echo "To check again:"
echo "  bash monitor_step1.sh"
