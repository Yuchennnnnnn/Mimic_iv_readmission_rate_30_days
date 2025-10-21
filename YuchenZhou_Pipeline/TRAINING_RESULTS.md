# 训练结果总结

## 🎯 训练配置

### 数据集
- **数据文件**: `cleaned_data.csv`
- **样本总数**: 205,980
- **训练集**: 164,784 样本 (80%)
- **测试集**: 41,196 样本 (20%)
- **再入院率**: 26.72%

### 特征选择
✅ **已启用特征选择** (基于Xi Chen的LASSO结果)

- **原始特征数**: 47列
- **LASSO筛选特征**: 121个 (来自 `Feature_Importance_by_Coef.csv`)
- **Top-N筛选**: 50个最重要特征
- **重要性阈值**: ≥ 0.05
- **最终使用特征**: 18个

#### Top 5 最重要特征:
1. **died_in_hospital** (0.6032) - 院内死亡
2. **last_service_OMED** (0.4505) - 最后服务类型
3. **gender_F** (0.3823) - 性别（女性）
4. **admission_type_SURGICAL SAME DAY ADMISSION** (0.3403) - 入院类型
5. **discharge_location_HOSPICE** (0.3182) - 出院地点

---

## 📊 模型性能 - Logistic Regression

### 训练详情
- **模型类型**: Logistic Regression (L2正则化)
- **特征维度**: 30 (包含One-Hot编码后的特征)
- **训练时间**: ~2分钟

### 评估指标

| 指标 | 数值 |
|------|------|
| **ROC-AUC** | 0.6626 |
| **PR-AUC** | 0.4037 |
| **Accuracy** | 0.5918 (59.18%) |
| **Precision** | 0.3576 (35.76%) |
| **Recall** | 0.6621 (66.21%) |
| **F1-Score** | 0.4643 (46.43%) |
| **Specificity** | 0.5662 (56.62%) |

### 混淆矩阵

|           | 预测: 不再入院 | 预测: 再入院 |
|-----------|----------------|--------------|
| **实际: 不再入院** | 17,093 (TN) | 13,095 (FP) |
| **实际: 再入院**   | 3,720 (FN)  | 7,288 (TP)  |

### 性能分析

**优点** ✅:
- 较高的召回率 (66.21%) - 能识别大部分真正会再入院的患者
- ROC-AUC = 0.66 - 比随机猜测 (0.5) 有明显提升
- 特征选择有效 - 仅用18个特征达到合理性能

**改进空间** ⚠️:
- 精确率较低 (35.76%) - 存在较多误报
- F1-Score = 0.46 - 精确率和召回率不平衡
- 类别不平衡问题 (73% vs 27%)

---

## 📁 生成的文件

### 模型文件 (`artifacts/`)
- `lr.pkl` - 训练好的Logistic Regression模型
- `lr_encoders.pkl` - One-Hot编码器
- `lr_scalers.pkl` - 数据标准化器

### 评估报告 (`reports/`)
- `metrics.csv` - 所有评估指标
- `predictions_lr.csv` - 测试集预测结果
- `roc_curve_lr.png` - ROC曲线图
- `pr_curve_lr.png` - Precision-Recall曲线图
- `confusion_matrix_lr.png` - 混淆矩阵热力图
- `calibration_curve_lr.png` - 概率校准曲线
- `feature_importance_lr.png` - 特征重要性图

---

## 🚀 下一步操作

### 1. 查看可视化结果
```bash
cd YuchenZhou_Pipeline/training/reports
open roc_curve_lr.png
open confusion_matrix_lr.png
open feature_importance_lr.png
```

### 2. 训练更多模型

#### 快速训练 Random Forest
```bash
cd YuchenZhou_Pipeline/training
python src/train.py --model rf --config config.yaml
```

#### 快速训练 XGBoost
```bash
python src/train.py --model xgb --config config.yaml
```

#### 训练所有传统模型（推荐）
```bash
python src/train.py --model logistic,rf,xgb --config config.yaml
```

#### 训练所有模型（包括深度学习，耗时约1小时）
```bash
python src/train.py --model all --config config.yaml
```

### 3. 使用训练好的模型进行预测

```bash
cd ../testing
python src/inference.py \
  --model ../training/artifacts/lr.pkl \
  --data ../../cleaned_data.csv \
  --output predictions.csv
```

### 4. 调整特征选择参数

编辑 `config.yaml`:

```yaml
feature_selection:
  enabled: true
  method: "importance_file"
  top_n: 100  # 增加特征数量
  importance_threshold: 0.01  # 降低阈值
```

然后重新训练：
```bash
python src/train.py --model logistic --config config.yaml
```

### 5. 模型对比与分析

训练多个模型后，查看对比结果：
- `reports/metrics.csv` - 包含所有模型的指标
- `reports/model_comparison.png` - 模型对比可视化

---

## 💡 建议

### 提升模型性能
1. **处理类别不平衡**:
   - 在 `config.yaml` 中启用 SMOTE: `use_smote: true`
   - 调整类别权重: `class_weight: balanced`

2. **调整决策阈值**:
   - 当前默认阈值 0.5
   - 可调整到 0.3-0.4 以提高召回率
   - 或调整到 0.6-0.7 以提高精确率

3. **增加特征**:
   - 尝试 `top_n: 100` 使用更多特征
   - 或设置 `enabled: false` 使用所有特征对比

4. **尝试更强大的模型**:
   - Random Forest - 通常性能更好
   - XGBoost - 处理不平衡数据效果好
   - 深度学习模型 - 可能捕获复杂模式

### 模型部署
1. 使用 `testing/src/inference.py` 进行批量预测
2. 模型文件在 `artifacts/` 目录下
3. 可直接用于新数据预测

---

## 📝 技术细节

### 预处理流程
1. **数据加载**: 从CSV读取MIMIC-IV数据
2. **特征选择**: 基于LASSO系数筛选top-50特征
3. **特征映射**: 将One-Hot编码名称映射回原始列名
4. **缺失值处理**: 
   - 类别特征: 填充"Unknown"
   - 数值特征: 填充中位数
5. **编码**: 
   - One-Hot编码: gender, marital_status, insurance, admission_type
   - 高基数特征删除: last_service, language, admission_location, discharge_location
6. **标准化**: StandardScaler归一化数值特征

### 模型参数
- **Solver**: lbfgs
- **Penalty**: L2正则化
- **Max Iterations**: 1000
- **Class Weight**: Balanced (自动处理类别不平衡)

---

## 🎓 项目信息

- **课程**: CS526 - Fall 2025
- **项目**: 30天医院再入院预测
- **数据集**: MIMIC-IV
- **学生**: Yuchen Zhou
- **合作**: 基于Xi Chen的LASSO特征选择结果

---

生成时间: 2025-01-XX
Pipeline版本: v1.0
