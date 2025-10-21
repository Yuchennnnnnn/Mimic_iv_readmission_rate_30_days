#!/bin/bash
# 快速训练脚本 - Yuchen Zhou's Readmission Prediction Pipeline

echo "=========================================="
echo "  30-Day Readmission Prediction"
echo "  Yuchen Zhou's Pipeline"
echo "=========================================="
echo ""

# 设置虚拟环境Python路径
VENV_PYTHON="/Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2/.venv/bin/python"

# 检查虚拟环境
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 错误: 虚拟环境未找到"
    echo "   请先创建虚拟环境: python -m venv .venv"
    exit 1
fi

# 进入训练目录
cd "$(dirname "$0")"

echo "选择训练模式:"
echo ""
echo "  1. 快速测试 (Logistic Regression only, ~2分钟)"
echo "  2. 传统ML模型 (LR + RF + XGBoost, ~15分钟)"
echo "  3. 所有模型 (包括LSTM和Transformer, ~1小时)"
echo "  4. 自定义"
echo "  5. 退出"
echo ""
read -p "请选择 [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 开始快速测试..."
        $VENV_PYTHON src/train.py --model logistic --config config.yaml
        ;;
    2)
        echo ""
        echo "🚀 训练传统ML模型..."
        $VENV_PYTHON src/train.py --model logistic --config config.yaml
        $VENV_PYTHON src/train.py --model rf --config config.yaml
        $VENV_PYTHON src/train.py --model xgb --config config.yaml
        ;;
    3)
        echo ""
        echo "🚀 训练所有模型（这需要一段时间）..."
        $VENV_PYTHON src/train.py --model all --config config.yaml
        ;;
    4)
        echo ""
        echo "可用模型: logistic, rf, xgb, lstm, transformer, all"
        read -p "输入模型名称: " model_name
        echo ""
        echo "🚀 训练 $model_name ..."
        $VENV_PYTHON src/train.py --model "$model_name" --config config.yaml
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ 训练成功完成！"
    echo "=========================================="
    echo ""
    echo "📊 查看结果:"
    echo "   指标: reports/metrics.csv"
    echo "   可视化: reports/*.png"
    echo "   模型: artifacts/*.pkl"
    echo ""
    echo "📖 详细说明: ../TRAINING_RESULTS.md"
else
    echo ""
    echo "=========================================="
    echo "  ❌ 训练失败"
    echo "=========================================="
    exit 1
fi
