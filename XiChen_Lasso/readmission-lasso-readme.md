# 医院再入院预测 - LASSO特征选择

## 项目简介
使用L1正则化逻辑回归（LASSO）预测患者30天内再入院风险，自动筛选关键特征。

## 数据集
- **来源**: MIMIC-III清洗后数据 (cleaned_data.csv)
- **目标变量**: readmit_label (30天内是否再入院)
- **特征数量**: 121个原始特征 → 102个被选中

## 核心结果

### Top 10 重要特征
| 特征名称 | 系数 | 含义 |
|---------|------|------|
| died_in_hospital | -0.6032 | 院内死亡 |
| last_service_OMED | +0.4505 | 急诊观察科 |
| gender_F | -0.3823 | 女性 |
| admission_type_SURGICAL SAME DAY | -0.3403 | 当日手术入院 |
| discharge_location_HOSPICE | -0.3182 | 出院至临终关怀 |
| last_service_ORTHO | -0.2996 | 骨科 |
| days_since_prev_discharge | -0.2812 | 距上次出院天数 |
| gender_M | -0.2198 | 男性 |
| insurance_Private | -0.2107 | 商业保险 |
| admission_location_TRANSFER FROM HOSPITAL | -0.2098 | 医院转入 |

## 模型配置
```python
LogisticRegression(
    penalty='l1',
    solver='saga',
    C=0.1,  # 正则化强度
    max_iter=500
)
```

## 代码执行流程
1. **数据加载**: 从Google Drive读取cleaned_data.csv
2. **预处理**: 数值特征标准化，分类特征独热编码
3. **模型训练**: L1正则化自动特征选择
4. **结果导出**: selected_features.csv包含所有非零系数特征

## 关键发现
- **降低再入院风险**: 女性、当日手术、商业保险
- **增加再入院风险**: 急诊科患者（OMED服务）
- **特殊情况**: 院内死亡和临终关怀患者系数为负（符合逻辑）

## 超参数分析
通过交叉验证测试C值（0.01-10），平衡特征数量与AUC性能。

## 输出文件
- `selected_features.csv`: 102个重要特征及系数值

## 下一步工作
- 尝试Elastic Net处理相关特征
- 添加时间验证评估模型稳定性
- 设定风险分层阈值用于临床应用