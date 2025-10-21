# 特征选择说明 - 为什么50个LASSO特征变成了18个原始列？

## 📊 快速回答

**这是正常的！** LASSO在One-Hot编码后的数据上训练，而我们使用的是编码前的原始数据。

- ✅ **LASSO选择**: 48个特征（One-Hot编码后）
- ✅ **映射结果**: 18个原始数据列
- ✅ **原因**: 多个编码特征对应同一个原始列

---

## 🔍 详细解释

### 1. LASSO特征选择过程

Xi Chen的LASSO模型：
```
原始数据 → One-Hot编码 → LASSO训练 → 选择重要特征
```

例如 `gender` 列：
```
原始: gender = ['M', 'F']
↓ One-Hot编码
编码后: gender_M = [1, 0]
       gender_F = [0, 1]
↓ LASSO选择
选择: gender_F (重要性 0.38), gender_M (重要性 0.22)
```

### 2. 我们的映射过程

我们的pipeline：
```
LASSO特征(48个) → 映射到原始列 → 原始数据(18列)
```

映射规则：
```python
'gender_F' + 'gender_M' → 'gender' (1个原始列)
'last_service_OMED' + 'last_service_ORTHO' + ... → 'last_service' (1个原始列)
```

---

## 📋 完整映射表

### 一对一映射（10个列）
这些列没有被One-Hot编码，直接匹配：

| 原始列 | LASSO特征 | 重要性 |
|--------|-----------|--------|
| `died_in_hospital` | died_in_hospital | 0.6032 ⭐ |
| `days_since_prev_discharge` | days_since_prev_discharge | 0.2812 |
| `anchor_age` | anchor_age | 0.1705 |
| `Hemoglobin_median` | Hemoglobin_median | 0.1066 |
| `num_diagnoses` | num_diagnoses | 0.0928 |
| `Hemoglobin_min` | Hemoglobin_min | 0.0865 |
| `unique_careunits` | unique_careunits | 0.0757 |
| `Glucose_median` | Glucose_median | 0.0705 |
| `is_surgical_service` | is_surgical_service | 0.0588 |
| `Potassium_min` | Potassium_min | 0.0578 |

### 一对多映射（8个列）
这些列被One-Hot编码，多个LASSO特征映射到1个原始列：

#### 1. `last_service` ← 7个LASSO特征
```
✓ last_service_OMED        (0.4505) ⭐
✓ last_service_ORTHO       (0.2996)
✓ last_service_NMED        (0.1572)
✓ last_service_CMED        (0.1119)
✓ last_service_CSURG       (0.0760)
✓ last_service_PSYCH       (0.0736)
✓ last_service_MED         (0.0637)
```

#### 2. `discharge_location` ← 7个LASSO特征
```
✓ discharge_location_HOSPICE                  (0.3182)
✓ discharge_location_HOME                     (0.1577)
✓ discharge_location_PSYCH FACILITY           (0.1414)
✓ discharge_location_DIED                     (0.1221)
✓ discharge_location_REHAB                    (0.1066)
✓ discharge_location_SKILLED NURSING FACILITY (0.1035)
✓ discharge_location_AGAINST ADVICE           (0.0706)
```

#### 3. `admission_type` ← 6个LASSO特征
```
✓ admission_type_SURGICAL SAME DAY ADMISSION  (0.3403)
✓ admission_type_OBSERVATION ADMIT            (0.2018)
✓ admission_type_URGENT                       (0.1462)
✓ admission_type_ELECTIVE                     (0.1309)
✓ admission_type_EW EMER.                     (0.1215)
✓ admission_type_DIRECT EMER.                 (0.1046)
```

#### 4. `marital_status` ← 4个LASSO特征
```
✓ marital_status_MARRIED   (0.2084)
✓ marital_status_SINGLE    (0.1470)
✓ marital_status_WIDOWED   (0.1449)
✓ marital_status_DIVORCED  (0.0932)
```

#### 5. `admission_location` ← 4个LASSO特征
```
✓ admission_location_TRANSFER FROM HOSPITAL   (0.2098)
✓ admission_location_WALK-IN/SELF REFERRAL    (0.1243)
✓ admission_location_PHYSICIAN REFERRAL       (0.0999)
✓ admission_location_CLINIC REFERRAL          (0.0523)
```

#### 6. `insurance` ← 4个LASSO特征
```
✓ insurance_Private   (0.2107)
✓ insurance_Medicare  (0.1700)
✓ insurance_Medicaid  (0.1372)
✓ insurance_Other     (0.0761)
```

#### 7. `language` ← 4个LASSO特征
```
✓ language_English  (0.1936)
✓ language_Spanish  (0.1011)
✓ language_Russian  (0.0762)
✓ language_Chinese  (0.0602)
```

#### 8. `gender` ← 2个LASSO特征
```
✓ gender_F  (0.3823)
✓ gender_M  (0.2198)
```

---

## 📈 数据降维效果

```
原始数据: 47 列
    ↓
LASSO筛选 (One-Hot编码后): 121 → 48 个重要特征
    ↓
映射回原始列: 18 列
    ↓
最终训练: 18 个特征列 + 3 个ID/Label列 = 21 列
```

**降维效果**: 47 → 18 列（减少 61.7%）

---

## 💡 为什么这样做？

### 优势
1. ✅ **利用LASSO结果**: 保留了Xi Chen发现的重要特征
2. ✅ **避免数据泄露**: 使用原始categorical列，让模型自己学习编码
3. ✅ **灵活性**: 不同模型可以用不同的编码方式
   - Logistic Regression: One-Hot编码
   - Random Forest: Label编码或直接使用
   - XGBoost: 直接处理categorical
4. ✅ **减少特征数**: 从47列减少到18列，训练更快

### 示例
比如 `gender` 列：
- **LASSO方式**: 选择 `gender_F` 和 `gender_M` 两个二值特征
- **我们的方式**: 保留 `gender` 一个列，让模型决定如何编码
  - LR会自动One-Hot编码成 gender_F, gender_M
  - RF可以直接使用categorical
  - XGBoost可以原生处理

---

## 🔧 如何调整特征数量？

如果你想使用更多特征，可以修改 `training/config.yaml`:

```yaml
feature_selection:
  enabled: true
  top_n: 100        # 增加到100 (当前: 50)
  importance_threshold: 0.01  # 降低阈值 (当前: 0.05)
```

预期效果：
- `top_n: 100` → 约 30-35 个原始列
- `top_n: 121` (全部) → 约 40+ 个原始列

---

## 📊 当前模型性能

使用18个特征的结果：

| 模型 | ROC-AUC | Recall | F1-Score |
|------|---------|--------|----------|
| **XGBoost** | **0.7029** ⭐ | 68.46% | 0.4938 |
| Random Forest | 0.6933 | 62.84% | 0.4824 |
| Logistic Regression | 0.6626 | 66.21% | 0.4643 |

✅ **结论**: 18个精选特征已经取得了很好的效果！

---

## 🎯 总结

**50个LASSO特征 → 18个原始列是完全正常的**

原因：
1. LASSO在One-Hot编码数据上训练（121维）
2. 选择了48个重要的编码特征
3. 这些特征映射回原始数据时合并为18个base columns
4. 我们的模型在这18列上训练，效果很好

这种设计：
- ✅ 充分利用了LASSO的特征选择结果
- ✅ 保持了数据的原始格式
- ✅ 让不同模型使用最适合的编码方式
- ✅ 训练速度快，性能好

---

## 📖 相关文件

- 映射检查脚本: `training/check_feature_mapping.py`
- 特征选择代码: `training/src/feature_selection.py`
- LASSO结果: `Feature_Importance_by_Coef.csv`
- 配置文件: `training/config.yaml`
