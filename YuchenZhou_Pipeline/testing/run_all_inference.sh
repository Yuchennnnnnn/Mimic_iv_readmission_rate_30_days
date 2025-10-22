#!/bin/bash
# Run inference for all trained models

echo "=========================================="
echo "  Batch Model Inference"
echo "  Predict using all trained models"
echo "=========================================="
echo ""

# Set virtual environment Python path
VENV_PYTHON="/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/.venv/bin/python"

# Enter testing directory
cd "$(dirname "$0")"

# Check data file
if [ ! -f "../../cleaned_data.csv" ]; then
    echo "❌ Error: Data file does not exist"
    exit 1
fi

echo "Data file: ../../cleaned_data.csv"
echo ""

# Define model list
declare -A models=(
    ["Logistic Regression"]="lr"
    ["Random Forest"]="rf"
    ["XGBoost"]="xgb"
)

# Statistics
total_models=0
success_models=0
failed_models=0

# Run inference for each model
for model_name in "${!models[@]}"; do
    model_short="${models[$model_name]}"
    model_path="../training/artifacts/${model_short}.pkl"
    preprocessor_path="../training/artifacts/${model_short}_preprocessor.joblib"
    output_path="reports/predictions_${model_short}.csv"
    
    total_models=$((total_models + 1))
    
    echo "----------------------------------------"
    echo "[$total_models/3] ${model_name}"
    echo "----------------------------------------"
    
    # Check model files
    if [ ! -f "$model_path" ]; then
        echo "  ⚠️  Model file does not exist, skipping"
        failed_models=$((failed_models + 1))
        echo ""
        continue
    fi
    
    echo "  Model: ${model_path}"
    echo "  Preprocessor: ${preprocessor_path}"
    echo "  Output: ${output_path}"
    echo ""
    
    # Run inference
    $VENV_PYTHON src/inference.py \
        --model-path "$model_path" \
        --preprocessor-path "$preprocessor_path" \
        --input ../../cleaned_data.csv \
        --output "$output_path" \
        --model-type sklearn 2>&1 | grep -E "(Loading|Loaded|Making|Saved|summary|Total|Predicted|Mean|range)"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "  ✅ Success"
        success_models=$((success_models + 1))
        
        # Display first few prediction results
        if [ -f "$output_path" ]; then
            echo ""
            echo "  Prediction results example (first 5 rows):"
            head -n 6 "$output_path" | tail -n 5 | column -t -s,
        fi
    else
        echo "  ❌ Failed"
        failed_models=$((failed_models + 1))
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "  Inference Completion Summary"
echo "=========================================="
echo ""
echo "  Total models: $total_models"
echo "  Successful: $success_models"
echo "  Failed: $failed_models"
echo ""

# Display all generated files
if [ $success_models -gt 0 ]; then
    echo "📊 Generated prediction files:"
    ls -lh reports/predictions_*.csv 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    
    # Comparison statistics
    echo "📈 Prediction Statistics Comparison by Model:"
    echo "----------------------------------------"
    printf "%-20s %15s %15s\n" "Model" "Predicted Readm" "Prediction Ratio"
    echo "----------------------------------------"
    
    for model_name in "${!models[@]}"; do
        model_short="${models[$model_name]}"
        output_path="reports/predictions_${model_short}.csv"
        
        if [ -f "$output_path" ]; then
            stats=$($VENV_PYTHON -c "
import pandas as pd
df = pd.read_csv('$output_path')
total = len(df)
positive = df['predicted_label'].sum()
ratio = positive / total * 100
print(f'{positive},{ratio:.1f}%')
" 2>/dev/null)
            
            if [ $? -eq 0 ]; then
                positive=$(echo $stats | cut -d',' -f1)
                ratio=$(echo $stats | cut -d',' -f2)
                printf "%-20s %15s %15s\n" "$model_name" "$positive" "$ratio"
            fi
        fi
    done
    echo "----------------------------------------"
fi

echo ""
echo "✅ All inference tasks completed!"
