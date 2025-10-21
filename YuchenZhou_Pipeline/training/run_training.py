#!/usr/bin/env python3
"""
简单的训练测试脚本 - 使用特征选择
"""

import os
import sys

print("="*80)
print("  Yuchen Zhou's 30-Day Readmission Prediction Pipeline")
print("  使用预计算的特征重要性进行训练")
print("="*80)
print()

# 检查环境
print("[1/5] 检查环境...")
if not os.path.exists('src/train.py'):
    print("❌ 错误: 请从 YuchenZhou_Pipeline/training 目录运行此脚本")
    sys.exit(1)

print("✓ 目录正确")

# 检查数据文件
print("\n[2/5] 检查数据文件...")
config_path = 'config.yaml'

import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = config['data']['input_path']
feat_imp_path = config['data']['feature_importance_path']

if not os.path.exists(data_path):
    print(f"❌ 错误: 数据文件不存在: {data_path}")
    print("   请更新 config.yaml 中的 input_path")
    sys.exit(1)

print(f"✓ 数据文件找到: {data_path}")

if not os.path.exists(feat_imp_path):
    print(f"⚠️  警告: 特征重要性文件不存在: {feat_imp_path}")
    print("   将使用所有特征进行训练")
else:
    print(f"✓ 特征重要性文件找到: {feat_imp_path}")

# 检查依赖
print("\n[3/5] 检查Python依赖...")
required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib', 'seaborn', 'yaml', 'joblib']
missing_packages = []

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'yaml':
            __import__('yaml')
        else:
            __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"❌ 缺少以下包: {', '.join(missing_packages)}")
    print("\n   请安装:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("✓ 所有依赖已安装")

# 显示配置
print("\n[4/5] 训练配置:")
print(f"   数据路径: {data_path}")
print(f"   特征选择: {'启用' if config.get('feature_selection', {}).get('enabled') else '禁用'}")
if config.get('feature_selection', {}).get('enabled'):
    top_n = config['feature_selection'].get('top_n', 'All')
    threshold = config['feature_selection'].get('importance_threshold', 'None')
    print(f"   - Top N特征: {top_n}")
    print(f"   - 重要性阈值: {threshold}")
print(f"   要训练的模型: {', '.join(config.get('models_to_run', []))}")

# 询问用户
print("\n[5/5] 准备训练...")
print("\n选择训练选项:")
print("  1. 快速测试 (仅Logistic Regression, ~2分钟)")
print("  2. 训练传统模型 (LR + RF + XGBoost, ~10分钟)")
print("  3. 训练所有模型 (包括深度学习, ~1小时)")
print("  4. 退出")

choice = input("\n请选择 [1-4]: ").strip()

if choice == '1':
    model_choice = 'logistic'
    print("\n开始快速测试...")
elif choice == '2':
    model_choice = 'logistic,rf,xgb'
    print("\n开始训练传统模型...")
elif choice == '3':
    model_choice = 'all'
    print("\n开始训练所有模型...")
elif choice == '4':
    print("退出")
    sys.exit(0)
else:
    print("无效选择，使用默认: 快速测试")
    model_choice = 'logistic'

# 运行训练
print("\n" + "="*80)
print("开始训练...")
print("="*80 + "\n")

import subprocess

if model_choice == 'all':
    cmd = ['python', 'src/train.py', '--model', 'all', '--config', config_path]
else:
    # 对于多个模型，依次训练
    models = model_choice.split(',')
    for model in models:
        print(f"\n>>> 训练 {model.upper()} <<<\n")
        cmd = ['python', 'src/train.py', '--model', model.strip(), '--config', config_path]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ {model} 训练失败")
            continue
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    sys.exit(0)

# 对于 'all'，直接运行
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ 训练成功完成!")
    print("="*80)
    print("\n查看结果:")
    print("  - 指标: reports/metrics.csv")
    print("  - 图表: reports/model_comparison.png")
    print("  - 模型: artifacts/")
else:
    print("\n" + "="*80)
    print("❌ 训练过程中出现错误")
    print("="*80)
    sys.exit(1)
