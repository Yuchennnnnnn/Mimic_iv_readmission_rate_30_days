# 快速参考 - Yuchen Zhou的30天再入院预测Pipeline# 🚀 QUICK REFERENCE CARD



## 🎯 最重要的结果## Installation & Setup (One Time)

```bash

| 指标 | 最佳模型 | 分数 |cd training

|------|---------|------|pip install -r requirements.txt

| **ROC-AUC** (推荐指标) | Transformer | 0.7056 |```

| **F1-Score** (平衡指标) | XGBoost | 0.4947 |

| **Recall** (捕获率) | XGBoost | 68.5% |## Quick Test (2 minutes)

```bash

**⭐ 推荐生产使用**: **XGBoost** (最佳平衡性能)cd training

python quick_start.py

---```



## 📊 所有模型对比## Train Models



| 模型 | ROC-AUC | Recall | F1 | 训练时间 |### Train All Models

|------|---------|--------|-----|---------|```bash

| **XGBoost** ⭐ | 0.7040 | 68.5% | 0.4947 | ~8分钟 |cd training

| **Transformer** | 0.7056 | 12.6% | 0.2104 | ~45分钟 |python src/train.py --model all --config config.yaml

| **LSTM** | 0.7030 | 14.5% | 0.2354 | ~40分钟 |```

| **Random Forest** | 0.6941 | 62.6% | 0.4834 | ~5分钟 |

| **Logistic Reg** | 0.6626 | 66.2% | 0.4643 | ~2分钟 |### Train Individual Models

```bash

---# Fastest (1 min)

python src/train.py --model logistic

## 🚀 一键运行命令

# Best performance (5 min)

### 快速训练（推荐）python src/train.py --model xgb

```bash

cd YuchenZhou_Pipeline/training# With custom epochs

./quick_train.shpython src/train.py --model lstm --epochs 30 --batch-size 64

# 选择 2 (训练LR + RF + XGBoost，15分钟)```

```

## View Results

### 单独训练最佳模型```bash

```bash# Metrics table

cd YuchenZhou_Pipeline/trainingcat reports/metrics.csv

python src/train.py --model xgb --config config.yaml

```# Open plots (macOS)

open reports/model_comparison.png

### 查看结果open reports/roc_curve_xgb.png

```bash

cd YuchenZhou_Pipeline/training/reports# Open plots (Linux)

cat metrics.csvxdg-open reports/model_comparison.png

open roc_curve_xgb.png```

```

## Make Predictions

---```bash

cd testing

## 📁 关键文件位置python src/inference.py \

  --model-path ../training/artifacts/xgb.pkl \

```  --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \

YuchenZhou_Pipeline/  --input new_data.csv \

├── README.md                          # 完整文档  --output predictions.csv \

├── FEATURE_SELECTION_EXPLANATION.md   # 为什么18个特征？  --model-type sklearn

├── training/```

│   ├── artifacts/xgb.pkl             # ⭐ 最佳模型

│   ├── reports/metrics.csv           # 所有结果## Configuration

│   └── config.yaml                   # 配置文件

└── Feature_Importance_by_Coef.csv    # LASSO特征重要性### Update Data Path

``````yaml

# In config.yaml

---data:

  input_path: "../cleaned_data.csv"  # Your data here

## 💡 关键发现```



### 1. 特征选择效果### Adjust Hyperparameters

- **原始**: 47列```yaml

- **LASSO筛选**: 48个重要特征（One-Hot编码后）# For imbalanced data

- **映射回原始**: 18列hyperparameters:

- **效果**: 降维61.7%，性能保持优秀  xgb:

    scale_pos_weight: 5.0  # Higher for more imbalance

### 2. 模型选择建议

# For faster training

**如果需要...**  lstm:

- **最高AUC**: 用Transformer (0.7056)    num_epochs: 20      # Reduce from 50

- **最好平衡**: 用XGBoost (F1=0.49, Recall=68%)  ⭐    batch_size: 128     # Increase from 64

- **快速原型**: 用Logistic Regression (~2分钟)```

- **可解释性**: 用Logistic Regression (系数清晰)

- **低误报**: 用Transformer (Precision=65%)## Troubleshooting



**生产环境**: **XGBoost是最佳选择！**### CUDA Out of Memory

```yaml

### 3. Top 5最重要特征# Reduce batch size in config.yaml

hyperparameters:

1. **died_in_hospital** (0.603) - 院内死亡  lstm:

2. **last_service_OMED** (0.451) - 服务类型    batch_size: 32

3. **gender** (0.382) - 性别```

4. **admission_type** (0.340) - 入院类型

5. **discharge_location** (0.318) - 出院地点### Import Errors

```bash

---# Run from training directory

cd training

## 🔧 快速调整python src/train.py ...

```

### 增加更多特征

```yaml### Slow Training

# 编辑 training/config.yaml```bash

feature_selection:# Use CPU only

  top_n: 100  # 改为100（当前50）export CUDA_VISIBLE_DEVICES=""

```

# Or select GPU

### 调整模型参数export CUDA_VISIBLE_DEVICES=0

```yaml```

# XGBoost超参数

models:## File Locations

  xgb:

    n_estimators: 300      # 更多树```

    max_depth: 7           # 更深的树training/

    learning_rate: 0.05    # 更小学习率├── src/train.py           ← Main training script

```├── config.yaml            ← Configuration

├── artifacts/             ← Saved models

---├── reports/               ← Results & plots

└── README.md              ← Full documentation

## 📈 性能解读

testing/

### XGBoost混淆矩阵├── src/inference.py       ← Prediction script

```└── artifacts/             ← Copy models here

实际 \ 预测    不再入院    再入院```

不再入院       18,244     11,944

再入院         3,465      7,543## Output Files

```

### After Training

**含义**:```

- ✅ 正确识别7,543个会再入院的患者artifacts/lr.pkl                    # Models

- ❌ 漏掉3,465个会再入院的患者 (31.5%)reports/metrics.csv                 # All metrics

- ⚠️ 误报11,944个不会再入院的患者reports/model_comparison.png        # Comparison chart

reports/roc_curve_*.png            # ROC curves

**适用场景**: 医院预防性干预（宁可误报也不漏报）reports/predictions_*.csv          # Predictions

```

---

## Key Commands Summary

## 🎓 论文写作要点

| Action | Command |

### 方法描述|--------|---------|

```| Quick test | `python quick_start.py` |

我们使用MIMIC-IV数据集的205,980个住院记录，| Train best model | `python src/train.py --model xgb` |

通过LASSO特征选择从47个原始特征中筛选出18个| Train all | `python src/train.py --model all` |

关键特征。训练了5个模型（LR, RF, XGBoost, | View metrics | `cat reports/metrics.csv` |

LSTM, Transformer），使用80/20分层划分。| Make predictions | `python ../testing/src/inference.py ...` |

XGBoost取得最佳综合性能（AUC=0.7040, | Run tests | `pytest tests/ -v` |

Recall=68.5%, F1=0.4947）。

```## Model Comparison Quick Reference



### 结果描述| Model | Speed | Performance | Use When |

```|-------|-------|-------------|----------|

所有模型AUC均超过0.66，深度学习模型（LSTM、| **Logistic** | ⚡⚡⚡ | ⭐⭐ | Need interpretability |

Transformer）达到0.70以上。XGBoost在召回率| **Random Forest** | ⚡⚡ | ⭐⭐⭐ | Want feature importance |

和F1分数上表现最佳，适合临床应用。特征重要性| **XGBoost** | ⚡⚡ | ⭐⭐⭐⭐ | **Best overall** |

分析显示院内死亡、服务类型、性别是最关键的| **LSTM** | ⚡ | ⭐⭐⭐ | Sequential patterns |

预测因子。| **Transformer** | ⚡ | ⭐⭐⭐ | Feature interactions |

```

## Typical Performance (MIMIC-IV)

---

```

## 🔍 常见问题Model            Time     ROC-AUC  F1-Score

──────────────────────────────────────────

**Q: 为什么选了50个LASSO特征只有18列？**  Logistic         1 min    0.67     0.38

A: LASSO在One-Hot编码数据上训练，多个编码特征（如gender_F, gender_M）对应同一个原始列（gender）。详见 [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md)Random Forest    5 min    0.71     0.41

XGBoost          5 min    0.73     0.43  ⭐

**Q: 哪个模型最好？**  LSTM            30 min    0.71     0.42

A: 看用途：Transformer     40 min    0.72     0.42

- **临床应用**: XGBoost（高recall，捕获68%再入院）```

- **风险排序**: Transformer（最高AUC）

- **快速部署**: Logistic Regression（训练快）## Support



**Q: 如何改进结果？**  📖 Full docs: `PIPELINE_README.md`  

A: 📖 Training guide: `training/README.md`  

1. 超参数调优（GridSearchCV）📖 Summary: `IMPLEMENTATION_SUMMARY.md`  

2. 增加更多特征（top_n: 100）

3. 集成学习（组合多个模型）---

4. 调整阈值（优化recall/precision）*Keep this file handy for quick reference!*


---

## 📞 快速帮助

**文件丢失了？**
```bash
cd YuchenZhou_Pipeline/training
python src/train.py --model xgb --config config.yaml
# 会重新生成所有文件
```

**环境有问题？**
```bash
pip install "numpy<2" pandas scikit-learn xgboost matplotlib seaborn pyyaml
```

**想看特征映射？**
```bash
python check_feature_mapping.py
```

---

## 📚 更多文档

- **完整README**: [README.md](./README.md)
- **特征选择解释**: [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md)
- **训练代码**: [training/src/train.py](./training/src/train.py)
- **配置文件**: [training/config.yaml](./training/config.yaml)

---

**最后更新**: 2025年10月  
**状态**: ✅ 所有模型已训练完成  
**数据**: 205,980样本，18特征，26.72%再入院率
