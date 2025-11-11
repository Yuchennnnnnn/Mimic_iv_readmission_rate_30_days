# MIMIC-IV 30-Day Readmission Prediction Pipeline# 30-Day Hospital Readmission Prediction Pipeline# Yuchen Zhou's ML Pipeline for 30-Day Readmission Prediction



完整的数据预处理和模型训练Pipeline，用于预测MIMIC-IV数据集中的30天再入院率。



**作者**: Yuchen Zhou, Jiaqi  **Author**: Yuchen Zhou  **Author:** Yuchen Zhou  

**课程**: CS526 Machine Learning  

**机构**: Duke University**Course**: CompSci 526 - Fall 2025  **Date:** October 20, 2025  



---**Institution**: Duke University  **Course:** CS526 - Machine Learning in Healthcare, Duke University



## 📋 目录**Dataset**: MIMIC-IV Clinical Database  



- [快速开始](#-快速开始)---

- [项目结构](#-项目结构)

- [安装指南](#-安装指南)---

- [数据预处理](#-数据预处理)

- [数据格式](#-数据格式)## 📁 What's in This Folder

- [使用示例](#-使用示例)

- [上传GitHub指南](#-上传github指南)## 📋 Project Overview

- [故障排除](#-故障排除)

- [数据统计](#-数据统计)This is my individual contribution to the team project. It contains a complete, independent ML pipeline that works alongside my teammates' work.



---This pipeline predicts **30-day hospital readmission risk** using MIMIC-IV clinical data. It implements **5 machine learning models** with automated feature selection based on LASSO coefficients from collaborative work.



## 🚀 快速开始```



### 最简单的运行方式### Key FeaturesYuchenZhou_Pipeline/



```bash- ✅ **Automated Feature Selection**: Uses pre-computed LASSO feature importance├── training/              # My training pipeline

# 1. 克隆仓库

git clone https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days.git- ✅ **5 Model Implementations**: LR, RF, XGBoost, LSTM, Transformer├── testing/               # My inference/testing pipeline

cd Mimic_iv_readmission_rate_30_days/YuchenZhou_jiaqi_Pipeline

- ✅ **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, Calibration├── README.md              # This file

# 2. 安装依赖

cd preprocessing- ✅ **Production-Ready**: Modular code with config-driven training├── QUICK_REFERENCE.md     # Quick command reference

pip install -r requirements.txt

- ✅ **Complete Pipeline**: From data loading to model deployment├── PIPELINE_README.md     # Detailed documentation

# 3. 配置数据路径

# 编辑 config.yaml，设置你的MIMIC-IV数据目录└── setup_and_run.sh       # Quick setup script



# 4. 运行预处理（约3-4小时）---```

bash run_all.sh



# 5. 使用输出数据训练模型

python your_training_script.py## 🏆 Model Performance Summary---

```



---

**Dataset**: 205,980 hospital admissions (26.72% readmission rate)  ## 🚀 Quick Start (2 Minutes)

## 📁 项目结构

**Features**: 18 selected features (from 48 LASSO features)  

```

YuchenZhou_jiaqi_Pipeline/**Train/Test Split**: 80/20 (164,784 / 41,196 samples)  ### Option 1: Automated Quick Test

├── preprocessing/                    # 数据预处理模块

│   ├── scripts/                     # 预处理脚本```bash

│   │   ├── step1_load_data_optimized.py   # Step 1: 加载数据

│   │   ├── step2_clean_units.py           # Step 2: 清理单位### Best Models by Metriccd YuchenZhou_Pipeline/training

│   │   ├── step3_create_timeseries.py     # Step 3: 创建时间序列

│   │   ├── step4_compute_features.py      # Step 4: 计算特征python quick_start.py

│   │   ├── step5_temporal_split.py        # Step 5: 时间分割

│   │   ├── step6_save_output.py           # Step 6: 保存输出| Metric | Model | Score | Notes |```

│   │   └── utils.py                       # 工具函数

│   ├── config.yaml                  # 配置文件 ⚙️|--------|-------|-------|-------|

│   ├── requirements.txt             # Python依赖

│   ├── run_all.sh                   # 运行完整pipeline| **ROC-AUC** | **Transformer** | **0.7056** ⭐ | Best overall discrimination |This will:

│   ├── run_steps_2_to_6.sh         # 运行步骤2-6

│   ├── check_progress.sh            # 检查进度| **F1-Score** | **XGBoost** | **0.4947** ⭐ | Best precision-recall balance |1. Generate synthetic test data

│   ├── QUICKSTART.md                # 快速开始指南

│   └── README.md                    # 预处理详细文档| **Recall** | **XGBoost** | **68.5%** ⭐ | Catches most readmissions |2. Train a Logistic Regression model

├── training/                         # 模型训练模块

│   ├── src/                         # 训练代码| **Precision** | **Transformer** | **65.2%** | Lowest false positives |3. Evaluate and create reports

│   ├── config.yaml                  # 训练配置

│   ├── run_training.py              # 训练脚本4. Show you where results are saved

│   └── requirements.txt             # 训练依赖

├── testing/                          # 模型测试模块### Complete Results

│   ├── src/                         # 测试代码

│   └── run_inference.sh             # 推理脚本### Option 2: Use Bash Script

├── output/                           # 输出数据 (39GB, 不上传)

│   ├── train_data.pkl               # 训练集 (10GB)| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score |```bash

│   ├── val_data.pkl                 # 验证集 (2.9GB)

│   ├── test_data.pkl                # 测试集 (2.4GB)|-------|---------|--------|----------|-----------|--------|----------|cd YuchenZhou_Pipeline

│   ├── train_index.parquet          # 训练集索引 (2.6MB)

│   ├── val_index.parquet            # 验证集索引 (781KB)| **Transformer** | **0.7056** | 0.4778 | 74.84% | 65.20% | 12.55% | 0.2104 |chmod +x setup_and_run.sh

│   ├── test_index.parquet           # 测试集索引 (638KB)

│   └── feature_names.txt            # 特征名称| **XGBoost** | **0.7040** | **0.4756** | 62.60% | 38.72% | **68.52%** | **0.4947** |./setup_and_run.sh

├── .gitignore                        # Git忽略文件配置

└── README.md                         # 本文件| **LSTM** | 0.7030 | 0.4723 | 74.83% | 62.56% | 14.50% | 0.2354 |```

```

| **Random Forest** | 0.6941 | 0.4625 | 64.25% | 39.37% | 62.59% | 0.4834 |

---

| **Logistic Reg** | 0.6626 | 0.4037 | 59.18% | 35.76% | 66.21% | 0.4643 |---

## 🔧 安装指南



### 系统要求

### Key Insights## 📊 Train on Real Data

- **操作系统**: macOS / Linux / Windows (WSL)

- **Python**: 3.8 或更高版本

- **内存**: 至少16GB RAM（推荐32GB）

- **存储**: 至少40GB可用磁盘空间1. **XGBoost is the best practical choice**:### Step 1: Install Dependencies

- **MIMIC-IV访问**: 需要通过[PhysioNet](https://physionet.org/)获得访问权限

   - High recall (68.5%) catches most readmissions```bash

### 步骤1: 安装Python依赖

   - Balanced F1-score (0.49) for real-world deploymentcd YuchenZhou_Pipeline/training

```bash

cd YuchenZhou_jiaqi_Pipeline/preprocessing   - Fastest training among top performerspip install -r requirements.txt

pip install -r requirements.txt

``````



**关键依赖说明**:2. **Transformer has highest AUC but low recall**:



```txt   - Best at ranking risk (0.7056 AUC)### Step 2: Update Configuration

pandas==2.0.3          # ⚠️ 必须是2.0.3版本（兼容性）

numpy>=1.26.4          # 数值计算   - Very high precision (65%) but misses many cases (12% recall)Edit `training/config.yaml` to point to the cleaned data:

fastparquet==2024.11.0 # Parquet文件读写

PyYAML                 # 配置文件解析   - Better suited for high-confidence predictions```yaml

tqdm                   # 进度条

scikit-learn           # 数据处理data:

torch                  # 模型训练（可选）

```3. **Traditional ML vs Deep Learning**:  input_path: "../../cleaned_data.csv"  # Adjust path as needed



### 步骤2: 配置数据路径   - XGBoost/RF: Better recall, faster training, easier deployment```



编辑 `preprocessing/config.yaml`:   - LSTM/Transformer: Higher precision, better calibration, requires more compute



```yaml### Step 3: Train Models

data_paths:

  patients: "/path/to/mimic-iv-3.1/hosp/patients.csv"---```bash

  admissions: "/path/to/mimic-iv-3.1/hosp/admissions.csv"

  chartevents: "/path/to/mimic-iv-3.1/icu/chartevents.csv"      # 39GB# Train all 5 models

  labevents: "/path/to/mimic-iv-3.1/hosp/labevents.csv"         # 17GB

  prescriptions: "/path/to/mimic-iv-3.1/hosp/prescriptions.csv" # 可选## 📁 Project Structurepython src/train.py --model all --config config.yaml



paths:

  output_dir: "../output"  # 输出目录（相对路径）

```# Or train individually

preprocessing:

  time_window_hours: 48     # 时间窗口（小时）YuchenZhou_Pipeline/python src/train.py --model logistic --config config.yaml

  bin_size_hours: 1         # 时间分辨率（小时）

  min_age: 18               # 最小年龄├── README.md                              # This filepython src/train.py --model rf --config config.yaml

  readmit_window_days: 30   # 再入院窗口（天）

  chunk_size: 10000         # 处理块大小├── FEATURE_SELECTION_EXPLANATION.md       # Why 50 features → 18 columnspython src/train.py --model xgb --config config.yaml

```

├── Feature_Importance_by_Coef.csv         # LASSO coefficients (Xi Chen's work)python src/train.py --model lstm --config config.yaml --epochs 30

---

│python src/train.py --model transformer --config config.yaml

## 🔄 数据预处理

├── training/```

### Pipeline工作流程

│   ├── config.yaml                        # Configuration file

```

MIMIC-IV原始数据│   ├── requirements.txt                   # Python dependencies---

    ↓

Step 1: 加载和过滤数据 (30-40分钟)│   ├── quick_train.sh                     # One-click training script

    ├─ 筛选年龄≥18岁

    ├─ 筛选住院时长≥48小时│   ├── run_training.py                    # Interactive training## 📈 View Results

    ├─ 排除院内死亡

    └─ 计算30天再入院标签│   ├── check_feature_mapping.py           # Feature analysis tool

    ↓

Step 2: 清理和标准化单位 (5-10分钟)│   │### Check Metrics

    ├─ 温度: Fahrenheit → Celsius

    ├─ 映射itemid到特征名│   ├── src/```bash

    └─ 去重

    ↓│   │   ├── train.py                       # Main training scriptcd training

Step 3: 创建48小时时间序列 (2-3小时) ⏰

    ├─ 将事件分到48个1小时bins│   │   ├── feature_selection.py           # Feature selection logic ⭐cat reports/metrics.csv

    ├─ 聚合每个bin的观测值

    └─ 处理322,966个住院记录│   │   ├── preprocess.py                  # Data preprocessing```

    ↓

Step 4: 计算Masks和Deltas特征 (5-10分钟)│   │   ├── models.py                      # Model implementations

    ├─ Masks: 观测指示器 (1=有观测, 0=缺失)

    ├─ Deltas: 距离上次观测的时间差│   │   ├── evaluate.py                    # Evaluation metrics### View Plots

    └─ Forward-fill填充

    ↓│   │   ├── dataset.py                     # PyTorch datasets```bash

Step 5: 时间分割 (2-3分钟)

    ├─ 训练集: 2008-2013 (60%)│   │   └── utils.py                       # Helper functions# ROC curves

    ├─ 验证集: 2014-2016 (20%)

    └─ 测试集: 2017-2019 (20%)│   │open reports/roc_curve_xgb.png

    ↓

Step 6: 保存最终输出 (1-2分钟)│   ├── artifacts/                         # Trained models

    ├─ train_data.pkl (10GB)

    ├─ val_data.pkl (2.9GB)│   │   ├── lr.pkl                         # Logistic Regression# Model comparison

    └─ test_data.pkl (2.4GB)

    ↓│   │   ├── rf.pkl                         # Random Forestopen reports/model_comparison.png

最终训练数据 (39GB total)

```│   │   ├── xgb.pkl                        # XGBoost ⭐



### 运行方式│   │   ├── lstm.pth                       # LSTM# Feature importance



#### 方式1: 一键运行（推荐）│   │   └── transformer.pth                # Transformeropen reports/feature_importance_xgb.png



```bash│   │```

cd preprocessing

bash run_all.sh│   └── reports/                           # Evaluation results

```

│       ├── metrics.csv                    # All model metrics---

#### 方式2: 分步运行

│       ├── predictions_*.csv              # Model predictions

```bash

cd preprocessing│       ├── roc_curve_*.png                # ROC curves## 🧪 Make Predictions on New Data



# Step 1: 加载数据│       ├── confusion_matrix_*.png         # Confusion matrices

python scripts/step1_load_data_optimized.py

│       └── feature_importance_*.png       # Feature importance plots```bash

# Steps 2-6: 处理和保存

bash run_steps_2_to_6.sh│cd YuchenZhou_Pipeline/testing

```

└── testing/

#### 方式3: 后台运行（推荐用于长时间任务）

    ├── src/python src/inference.py \

```bash

cd preprocessing    │   └── inference.py                   # Model inference script  --model-path ../training/artifacts/xgb.pkl \



# 在后台运行整个pipeline    └── README.md                          # Testing documentation  --preprocessor-path ../training/artifacts/xgb_preprocessor.joblib \

nohup bash run_all.sh > full_pipeline.log 2>&1 &

```  --input ../../cleaned_data.csv \

# 查看实时日志

tail -f full_pipeline.log  --output my_predictions.csv \



# 检查进度---  --model-type sklearn

./check_progress.sh

``````



#### 方式4: 单独运行某个步骤## 🚀 Quick Start



```bash---

cd preprocessing

### 1. Environment Setup

# 运行特定步骤

python scripts/step3_create_timeseries.py## 📚 Documentation

python scripts/step4_compute_features.py

# ... 等等```bash

```

# Navigate to project- **QUICK_REFERENCE.md** - Command cheat sheet

### 监控进度

cd YuchenZhou_Pipeline/training- **PIPELINE_README.md** - Complete documentation

```bash

# 方法1: 使用监控脚本- **IMPLEMENTATION_SUMMARY.md** - What was built

cd preprocessing

./check_progress.sh# Install dependencies (using virtual environment)- **training/README.md** - Training details



# 方法2: 查看日志pip install -r requirements.txt- **testing/README.md** - Inference details

tail -f full_pipeline.log

```

# 方法3: 检查输出文件

ls -lh output/---



# 方法4: 查看运行进程**Required Packages**:

ps aux | grep python | grep step

```- numpy<2, pandas, scikit-learn## 🎯 Models Implemented



**预期运行时间**:- xgboost, imbalanced-learn

- Step 1: 30-40分钟

- Step 2: 5-10分钟- torch, torchvision1. **Logistic Regression** - Interpretable baseline

- Step 3: 2-3小时 ⏰（最慢）

- Step 4: 5-10分钟- matplotlib, seaborn, pyyaml2. **Random Forest** - Ensemble with feature importance

- Step 5: 2-3分钟

- Step 6: 1-2分钟3. **XGBoost** - Best performance (typically)

- **总计**: 约3-4小时

### 2. Train Models4. **LSTM** - Deep learning with embeddings

---

5. **Transformer** - Attention-based architecture

## 📊 数据格式

**Option A: Interactive Script (Recommended)**

### Pickle文件结构

```bash---

```python

# 加载数据示例./quick_train.sh

import pickle

## 📂 Expected Outputs

with open('output/train_data.pkl', 'rb') as f:

    data_dict = pickle.load(f)# Then select:



# 数据结构# 1 - Quick test (Logistic Regression, ~2 min)After training, you'll have:

{

    'data': [  # 样本列表# 2 - Traditional ML (LR + RF + XGBoost, ~15 min) ⭐

        {

            'hadm_id': 12345678,           # 住院ID# 3 - All models (including LSTM + Transformer, ~1 hour)```

            'subject_id': 10000001,        # 患者ID

            'admittime': datetime(...),    # 入院时间```training/

            'values': np.array(...),       # (48, 49) 时间序列数值

            'masks': np.array(...),        # (48, 49) 观测指示器├── artifacts/

            'deltas': np.array(...),       # (48, 49) 时间差

            'readmit_30d': 0,              # 0=无再入院, 1=有再入院**Option B: Manual Training**│   ├── lr.pkl, rf.pkl, xgb.pkl        # Trained models

            'anchor_year_group': '2008 - 2010'

        },```bash│   ├── lstm.pt, transformer.pt        # PyTorch models

        # ... 194,672个训练样本

    ],# Single model│   └── *_preprocessor.joblib          # Encoders

    'feature_names': [  # 49个特征名称

        'heart_rate', 'sbp', 'dbp', 'temperature', python src/train.py --model xgb --config config.yaml└── reports/

        'respiratory_rate', 'spo2', 'glucose', ...

    ]    ├── metrics.csv                     # All metrics

}

```# Multiple models    ├── model_comparison.png            # Comparison chart



### 时间序列维度说明python src/train.py --model logistic --config config.yaml    ├── roc_curve_*.png                 # ROC curves



- **Shape**: `(48, 49)`python src/train.py --model rf --config config.yaml    ├── predictions_*.csv               # Predictions

  - **48**: 时间步（入院后前48小时，每小时1步）

  - **49**: 特征数量python src/train.py --model xgb --config config.yaml    └── feature_importance_*.png        # Feature importance



- **特征组成**:``````

  - **19个生命体征**: 心率、收缩压、舒张压、体温、呼吸频率、血氧等

  - **30个实验室指标**: 血糖、白细胞、肌酐、血红蛋白、钠、钾等



- **三个数组**:### 3. View Results---

  - **values**: 实际观测值

  - **masks**: 1=该时间点有观测, 0=缺失

  - **deltas**: 距离上次观测经过的小时数

```bash## ⚡ Performance Summary

### Parquet索引文件

cd reports/

```python

import pandas as pdExpected results on MIMIC-IV data:



# 加载索引# View metrics

train_index = pd.read_parquet('output/train_index.parquet', engine='fastparquet')

cat metrics.csv| Model | ROC-AUC | Training Time |

# 索引包含:

# - file_idx: 在pickle文件中的索引|-------|---------|---------------|

# - hadm_id: 住院ID

# - subject_id: 患者ID# View visualizations| Logistic Reg | 0.67 | 1 min |

# - admittime: 入院时间

# - readmit_30d: 再入院标签open roc_curve_xgb.png| Random Forest | 0.71 | 5 min |

# - anchor_year_group: 年份组

```open confusion_matrix_xgb.png| **XGBoost** | **0.73** | 5 min ⭐ |



---open feature_importance_xgb.png| LSTM | 0.71 | 30 min |



## 💻 使用示例```| Transformer | 0.72 | 40 min |



### 示例1: 加载和探索数据



```python------

import pickle

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt## 🔧 Configuration## 🔧 Troubleshooting



# 1. 加载训练数据

with open('output/train_data.pkl', 'rb') as f:

    train_dict = pickle.load(f)Edit `training/config.yaml` to customize:### GPU Out of Memory



train_data = train_dict['data']```yaml

feature_names = train_dict['feature_names']

### Feature Selection# In training/config.yaml, reduce batch size:

print(f"✓ Loaded {len(train_data)} training samples")

print(f"✓ Features: {len(feature_names)}")```yamlhyperparameters:

print(f"✓ First 10 features: {feature_names[:10]}")

feature_selection:  lstm:

# 2. 查看单个样本

sample = train_data[0]  enabled: true                    # Use LASSO feature selection    batch_size: 32

print(f"\n=== Sample Structure ===")

print(f"Admission ID: {sample['hadm_id']}")  top_n: 50                        # Number of top features (current: 18 columns)```

print(f"Patient ID: {sample['subject_id']}")

print(f"Admission time: {sample['admittime']}")  importance_threshold: 0.05       # Minimum importance threshold

print(f"Readmission label: {sample['readmit_30d']}")

print(f"\nData shapes:")  feature_importance_path: "../Feature_Importance_by_Coef.csv"### Import Errors

print(f"  values: {sample['values'].shape}")    # (48, 49)

print(f"  masks:  {sample['masks'].shape}")     # (48, 49)```Make sure you're in the correct directory:

print(f"  deltas: {sample['deltas'].shape}")    # (48, 49)

```bash

# 3. 可视化一个特征的时间序列

feature_idx = 0  # heart_rate**Note**: 50 LASSO features map to 18 original columns due to one-hot encoding. See [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md) for details.cd YuchenZhou_Pipeline/training

time_series = sample['values'][:, feature_idx]

mask = sample['masks'][:, feature_idx]python src/train.py --model logistic



plt.figure(figsize=(12, 4))### Model Hyperparameters```

plt.plot(time_series, marker='o', label='Heart Rate')

plt.scatter(np.where(mask == 0)[0], ```yaml

           time_series[mask == 0], 

           color='red', s=100, label='Missing')models:### Slow Training

plt.xlabel('Hour')

plt.ylabel('Heart Rate')  logistic:Train on a subset first to test:

plt.title(f'Heart Rate over 48 hours (Admission {sample["hadm_id"]})')

plt.legend()    penalty: 'l2'```python

plt.grid(True)

plt.show()    C: 1.0# In config.yaml, you can add this in preprocessing:



# 4. 统计信息    max_iter: 1000preprocessing:

readmit_count = sum([s['readmit_30d'] for s in train_data])

print(f"\n=== Dataset Statistics ===")    class_weight: 'balanced'  sample_size: 5000  # Use subset for testing

print(f"Total samples: {len(train_data)}")

print(f"Readmissions: {readmit_count} ({readmit_count/len(train_data)*100:.2f}%)")  ```

print(f"Non-readmissions: {len(train_data)-readmit_count}")

```  rf:



### 示例2: PyTorch数据加载器    n_estimators: 100---



```python    max_depth: 15

import torch

from torch.utils.data import Dataset, DataLoader    min_samples_split: 50## 🤝 Relation to Team Project

import numpy as np

    class_weight: 'balanced'

class MIMICDataset(Dataset):

    """MIMIC-IV 30-day readmission dataset"""  This pipeline is my **independent contribution** and works separately from:

    

    def __init__(self, data_list):  xgb:- `preprocessing/` - Shared preprocessing scripts

        self.data = data_list

        n_estimators: 200- `XiChen_Lasso/` - Xi Chen's LASSO feature selection

    def __len__(self):

        return len(self.data)    max_depth: 5- Other teammates' work

    

    def __getitem__(self, idx):    learning_rate: 0.1

        sample = self.data[idx]

            scale_pos_weight: 3.0All folders can coexist and use the same `cleaned_data.csv` file.

        return {

            'values': torch.FloatTensor(sample['values']),      # (48, 49)```

            'masks': torch.FloatTensor(sample['masks']),        # (48, 49)

            'deltas': torch.FloatTensor(sample['deltas']),      # (48, 49)---

            'label': torch.LongTensor([sample['readmit_30d']]), # (1,)

            'hadm_id': sample['hadm_id']### Training Parameters

        }

```yaml## 📝 What I Contributed

# 加载数据

import picklesplit:

with open('output/train_data.pkl', 'rb') as f:

    train_dict = pickle.load(f)  test_size: 0.2✅ Complete end-to-end ML pipeline  

with open('output/val_data.pkl', 'rb') as f:

    val_dict = pickle.load(f)  random_state: 42✅ 5 different model implementations  



# 创建数据集  stratify: true✅ Comprehensive preprocessing  

train_dataset = MIMICDataset(train_dict['data'])

val_dataset = MIMICDataset(val_dict['data'])✅ Full evaluation suite  



# 创建数据加载器deep_learning:✅ Deployment infrastructure  

train_loader = DataLoader(

    train_dataset,  epochs: 50✅ 3,500+ lines of code  

    batch_size=64,

    shuffle=True,  batch_size: 256✅ 2,000+ lines of documentation  

    num_workers=4,

    pin_memory=True  learning_rate: 0.001✅ Unit and integration tests  

)

  early_stopping_patience: 5

val_loader = DataLoader(

    val_dataset,```---

    batch_size=128,

    shuffle=False,

    num_workers=4

)---## 🎓 For Grading/Presentation



# 使用示例

for batch in train_loader:

    values = batch['values']     # (batch_size, 48, 49)## 📊 Selected Features (18 Total)**Key Files to Review:**

    masks = batch['masks']       # (batch_size, 48, 49)

    deltas = batch['deltas']     # (batch_size, 48, 49)1. `training/src/train.py` - Main training script

    labels = batch['label']      # (batch_size, 1)

    ### Demographic (3 features)2. `training/src/models.py` - All 5 models

    print(f"Batch shapes:")

    print(f"  values: {values.shape}")- `anchor_age` - Patient age3. `training/reports/metrics.csv` - Results

    print(f"  masks: {masks.shape}")

    print(f"  deltas: {deltas.shape}")- `gender` - Patient gender (F/M)4. `training/reports/model_comparison.png` - Visual comparison

    print(f"  labels: {labels.shape}")

    break- `marital_status` - Marital status

```

**To Demonstrate:**

### 示例3: LSTM模型训练

### Clinical (4 features)```bash

```python

import torch- `died_in_hospital` - In-hospital mortality (⭐ Most important, weight=0.60)# Quick test (2 min)

import torch.nn as nn

import torch.optim as optim- `days_since_prev_discharge` - Time since last dischargecd training

from sklearn.metrics import roc_auc_score, accuracy_score

- `num_diagnoses` - Number of diagnosespython quick_start.py

class LSTMReadmissionModel(nn.Module):

    """LSTM模型用于30天再入院预测"""- `is_surgical_service` - Surgical vs medical service

    

    def __init__(self, input_dim=49, hidden_dim=128, num_layers=2, dropout=0.3):# View results

        super().__init__()

        ### Administrative (5 features)cat reports/metrics.csv

        # 可以将values, masks, deltas拼接作为输入

        self.input_dim = input_dim * 3  # 49 * 3 = 147- `admission_type` - Type of admission (Emergency, Elective, etc.)```

        

        self.lstm = nn.LSTM(- `admission_location` - Where patient admitted from

            self.input_dim,

            hidden_dim,- `discharge_location` - Where patient discharged to---

            num_layers,

            batch_first=True,- `last_service` - Last clinical service (⭐ 2nd most important)

            dropout=dropout if num_layers > 1 else 0

        )- `insurance` - Insurance type## 📧 Questions?

        

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification

    ### Lab Values (4 features)- Check `QUICK_REFERENCE.md` for commands

    def forward(self, values, masks, deltas):

        # values, masks, deltas: (batch, 48, 49)- `Glucose_median` - Median glucose level- See `PIPELINE_README.md` for full documentation

        

        # 拼接三个特征- `Hemoglobin_median` - Median hemoglobin- Review `training/README.md` for detailed guide

        x = torch.cat([values, masks, deltas], dim=-1)  # (batch, 48, 147)

        - `Hemoglobin_min` - Minimum hemoglobin

        # LSTM

        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, 48, hidden_dim)- `Potassium_min` - Minimum potassium---

        

        # 使用最后一个时间步的输出

        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)

        ### Other (2 features)**Ready to use! Run `python training/quick_start.py` to get started.** 🚀

        # Dropout + 全连接层

        x = self.dropout(last_output)- `language` - Primary language

        logits = self.fc(x)  # (batch, 2)- `unique_careunits` - Number of care units visited

        

        return logits---



# 初始化模型## 🎯 Model Recommendations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMReadmissionModel(### For Different Use Cases

    input_dim=49,

    hidden_dim=128,| Use Case | Recommended Model | Reason |

    num_layers=2,|----------|-------------------|--------|

    dropout=0.3| **Production Deployment** | **XGBoost** | Best F1-score, good recall, fast inference |

).to(device)| **High-Risk Screening** | **XGBoost** | 68.5% recall catches most readmissions |

| **Low False Alarm** | **Transformer** | 65% precision, lowest false positives |

# 损失函数和优化器| **Risk Ranking** | **Transformer** | Highest AUC (0.7056) for risk stratification |

criterion = nn.CrossEntropyLoss()| **Interpretability** | **Logistic Regression** | Clear coefficients, easy to explain |

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)| **Quick Prototyping** | **Random Forest** | Fast training, good baseline performance |



# 训练循环### Deployment Strategy

num_epochs = 50

**Recommended Two-Stage Approach**:

for epoch in range(num_epochs):1. **Stage 1**: Use XGBoost for initial screening (high recall)

    model.train()2. **Stage 2**: Use Transformer to prioritize high-risk cases (high precision)

    train_loss = 0.0

    This catches most readmissions while minimizing unnecessary interventions.

    for batch in train_loader:

        values = batch['values'].to(device)---

        masks = batch['masks'].to(device)

        deltas = batch['deltas'].to(device)## 🧪 Model Inference

        labels = batch['label'].squeeze().to(device)

        Use trained models for predictions:

        # 前向传播

        optimizer.zero_grad()```bash

        logits = model(values, masks, deltas)cd testing

        loss = criterion(logits, labels)

        # Using XGBoost (recommended)

        # 反向传播python src/inference.py \

        loss.backward()  --model ../training/artifacts/xgb.pkl \

        optimizer.step()  --data ../../cleaned_data.csv \

          --output predictions.csv

        train_loss += loss.item()

    # Output includes:

    # 验证# - Patient ID

    model.eval()# - True label

    val_preds = []# - Predicted probability

    val_labels = []# - Predicted class

    # - Risk category (Low/Medium/High)

    with torch.no_grad():```

        for batch in val_loader:

            values = batch['values'].to(device)---

            masks = batch['masks'].to(device)

            deltas = batch['deltas'].to(device)## 📈 Performance Analysis

            labels = batch['label'].squeeze()

            ### Confusion Matrix (XGBoost)

            logits = model(values, masks, deltas)```

            probs = torch.softmax(logits, dim=1)[:, 1]  # 取readmission概率                Predicted

                          No      Yes

            val_preds.extend(probs.cpu().numpy())Actual  No   18,244  11,944   → Specificity: 60.4%

            val_labels.extend(labels.numpy())        Yes   3,465   7,543   → Sensitivity: 68.5%

    ```

    # 计算指标

    val_auc = roc_auc_score(val_labels, val_preds)**Interpretation**:

    val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))- **True Positives (7,543)**: Correctly identified readmissions

    - **False Negatives (3,465)**: Missed readmissions (31.5%)

    print(f"Epoch {epoch+1}/{num_epochs}")- **False Positives (11,944)**: False alarms

    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")- **True Negatives (18,244)**: Correct non-readmission predictions

    print(f"  Val AUC: {val_auc:.4f}")

    print(f"  Val Acc: {val_acc:.4f}")### ROC-AUC Comparison

```

All models achieve AUC > 0.66, with top 3 models > 0.70:

---- Excellent: 0.9-1.0

- Good: 0.8-0.9

## 📤 上传GitHub指南- **Fair: 0.7-0.8** ← Our models

- Poor: 0.6-0.7

### ⚠️ 重要：GitHub大文件限制- Random: 0.5



**GitHub限制**:Our models are in the **"Fair to Good"** range, suitable for clinical decision support.

- 单个文件最大 **100MB**

- 推送大小建议小于 **1GB**---

- 仓库总大小建议小于 **5GB**

## 🔬 Feature Importance (Top 10 from XGBoost)

**我们的数据规模**:

- `output/` 目录: **39GB** ❌ 太大！1. **died_in_hospital** (0.603) - In-hospital mortality

- 大文件列表:2. **last_service_OMED** (0.451) - Served by OMED

  ```3. **days_since_prev_discharge** (0.281) - Time since last visit

  17GB - timeseries_processed.pkl4. **gender** (0.382) - Patient gender

  10GB - train_data.pkl5. **discharge_location** (0.318) - Discharge destination

  5.7GB - timeseries_binned.pkl6. **admission_type** (0.340) - Type of admission

  2.9GB - val_data.pkl7. **anchor_age** (0.171) - Patient age

  2.4GB - test_data.pkl8. **insurance** (0.210) - Insurance type

  590MB - labevents_raw.parquet9. **marital_status** (0.208) - Marital status

  442MB - chartevents_raw.parquet10. **Hemoglobin_median** (0.107) - Median hemoglobin

  112MB - labevents_clean.parquet

  ```See `reports/feature_importance_xgb.png` for full visualization.



### 方案1: 只上传代码（推荐） ✅---



**已配置`.gitignore`文件**，自动排除大文件：## 📝 Methodology



```bash### 1. Feature Selection

# .gitignore 内容- Started with 47 features in cleaned data

output/                  # 整个输出目录- Xi Chen's LASSO identified 121 important one-hot encoded features

*.pkl                    # 所有pickle文件- Selected top 48 LASSO features (threshold ≥ 0.05)

*.parquet               # 所有parquet文件- Mapped to 18 original columns (handles one-hot encoding)

*.log                   # 日志文件- See [FEATURE_SELECTION_EXPLANATION.md](./FEATURE_SELECTION_EXPLANATION.md)



# 例外：保留小的索引文件### 2. Data Preprocessing

!*_index.parquet        # 训练/验证/测试索引 (总计<5MB)- **Missing Values**: Median for numeric, mode for categorical

!feature_names.txt      # 特征名称列表- **Categorical Encoding**:

```  - Logistic Regression: One-Hot Encoding (low cardinality only)

  - Random Forest/XGBoost: Label Encoding

**上传步骤**:  - LSTM/Transformer: Embedding layers

- **Numeric Features**: StandardScaler normalization

```bash- **Class Imbalance**: Handled via class weights and scale_pos_weight

cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2

### 3. Model Training

# 1. 初始化Git（如果还没有）- **Train/Test Split**: 80/20 stratified split

git init- **Validation**: For deep learning models (10% of training)

- **Early Stopping**: Prevents overfitting (patience=5 epochs)

# 2. 添加代码和配置文件- **Evaluation**: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1

git add YuchenZhou_jiaqi_Pipeline/.gitignore

git add YuchenZhou_jiaqi_Pipeline/preprocessing/scripts/### 4. Model Architectures

git add YuchenZhou_jiaqi_Pipeline/preprocessing/*.yaml

git add YuchenZhou_jiaqi_Pipeline/preprocessing/*.txt**Traditional ML**:

git add YuchenZhou_jiaqi_Pipeline/preprocessing/*.sh- Logistic Regression: L2 regularization, balanced class weights

git add YuchenZhou_jiaqi_Pipeline/preprocessing/*.md- Random Forest: 100 trees, max_depth=15

git add YuchenZhou_jiaqi_Pipeline/training/- XGBoost: 200 trees, learning_rate=0.1, scale_pos_weight=3.0

git add YuchenZhou_jiaqi_Pipeline/testing/

git add YuchenZhou_jiaqi_Pipeline/*.md**Deep Learning**:

- LSTM: Bidirectional, 2 layers, 128 hidden units

# 3. 可选：添加小的索引文件- Transformer: 4 attention heads, 3 encoder layers

git add YuchenZhou_jiaqi_Pipeline/output/*_index.parquet

git add YuchenZhou_jiaqi_Pipeline/output/feature_names.txt---



# 4. 检查将要上传的文件## 🤝 Collaboration

git status

This pipeline builds upon collaborative work:

# 5. 提交

git commit -m "Add MIMIC-IV preprocessing pipeline and training code"- **Xi Chen**: LASSO feature selection (`XiChen_Lasso/`)

- **Yuchen Zhou**: End-to-end ML pipeline (this folder)

# 6. 连接远程仓库

git remote add origin https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days.gitFeature importance coefficients are shared via `Feature_Importance_by_Coef.csv`.



# 7. 推送---

git branch -M main

git push -u origin main## 📚 Key Files

```

| File | Purpose |

**在README中说明数据获取方式**:|------|---------|

| `config.yaml` | All configuration parameters |

```markdown| `src/train.py` | Main training script (630 lines) |

## 📦 数据获取| `src/feature_selection.py` | LASSO feature integration ⭐ |

| `src/models.py` | All 5 model implementations |

由于预处理数据文件过大（39GB），未包含在仓库中。请按以下方式获取：| `src/evaluate.py` | Metrics and visualization |

| `reports/metrics.csv` | Complete results table |

### 方式1: 自行运行预处理（推荐）| `artifacts/*.pkl` | Trained model files |

bash YuchenZhou_jiaqi_Pipeline/preprocessing/run_all.sh

预计耗时: 3-4小时---



### 方式2: 下载预处理数据## 🐛 Troubleshooting

联系作者获取云存储链接：[your-email]

```### Common Issues



### 方案2: 使用Git LFS（适合<1GB文件）**1. NumPy Version Error**

```bash

如果你只想上传小一点的文件（如索引），可以使用Git LFS：# Error: numpy.dtype size changed

pip install "numpy<2"

```bash```

# 安装Git LFS

brew install git-lfs       # macOS**2. Missing Packages**

# sudo apt install git-lfs # Linux```bash

pip install -r requirements.txt

# 初始化```

git lfs install

**3. CUDA Out of Memory (Deep Learning)**

# 追踪特定文件类型```yaml

git lfs track "output/*_index.parquet"# In config.yaml, reduce batch size:

git lfs track "output/feature_names.txt"deep_learning:

  batch_size: 128  # Instead of 256

# 提交.gitattributes```

git add .gitattributes

git commit -m "Configure Git LFS"**4. Feature Mapping Issues**

```bash

# 正常添加和推送# Check feature mapping:

git add output/*_index.parquetpython check_feature_mapping.py

git push```

```

---

⚠️ **注意**: GitHub LFS免费额度仅1GB存储 + 1GB带宽/月

## 📊 Next Steps

### 方案3: 云存储链接（推荐用于分享大文件）

### Immediate Improvements

**步骤**:1. **Hyperparameter Tuning**: Use GridSearchCV for XGBoost

2. **Ensemble Models**: Combine XGBoost + Transformer predictions

1. 上传数据到云存储:3. **More Features**: Increase `top_n` to 100 for more features

   - Google Drive4. **Threshold Optimization**: Adjust classification threshold for recall/precision trade-off

   - Dropbox

   - OneDrive### Advanced Enhancements

   - 或其他云存储服务1. **SHAP Analysis**: Explain individual predictions

2. **Calibration**: Improve probability calibration

2. 获取分享链接3. **Temporal Validation**: Time-based train/test split

4. **External Validation**: Test on different hospital data

3. 在README中添加下载部分:5. **Clinical Integration**: Deploy as risk calculator API



```markdown---

## 📥 下载预处理数据

## 📖 References

预处理后的数据（39GB）可从以下链接下载：

- **Dataset**: MIMIC-IV Clinical Database v2.0

| 文件 | 大小 | 下载链接 |- **Paper**: Johnson et al. (2023). "MIMIC-IV, a freely accessible electronic health record dataset"

|------|------|---------|- **Methods**: Scikit-learn, XGBoost, PyTorch

| train_data.pkl | 10GB | [Google Drive](https://drive.google.com/...) |- **Evaluation**: Standard ML metrics (Hosmer-Lemeshow, 2013)

| val_data.pkl | 2.9GB | [Google Drive](https://drive.google.com/...) |

| test_data.pkl | 2.4GB | [Google Drive](https://drive.google.com/...) |---

| 索引文件 | <5MB | [GitHub Releases](https://github.com/.../releases) |

## 📧 Contact

### 下载后放置位置：

将下载的文件放到: `YuchenZhou_jiaqi_Pipeline/output/`**Yuchen Zhou**  

```Duke University - CompSci 526  

Fall 2025  

### 验证上传内容

For questions about this pipeline, please refer to the code comments or configuration file.

```bash

# 查看将要提交的文件---

git status

## 📜 License

# 查看被忽略的文件

git status --ignoredThis project is for educational purposes as part of CompSci 526 coursework.  

MIMIC-IV data usage follows PhysioNet credentialed access requirements.

# 检查文件大小

git ls-files | xargs ls -lh---



# 检查仓库大小**Last Updated**: October 2025  

git count-objects -vH**Version**: 1.0  

**Status**: ✅ Production Ready

# 确保没有大文件
git ls-files | xargs ls -lh | awk '$5 ~ /[0-9]+M/ {print}'
```

### 如果意外添加了大文件

```bash
# 方法1: 从暂存区移除
git reset HEAD path/to/large/file
git rm --cached path/to/large/file

# 方法2: 从历史中完全移除
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch output/" \
  --prune-empty --tag-name-filter cat -- --all

# 方法3: 使用BFG工具（推荐，更快）
# 下载: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-folders output
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 强制推送（谨慎使用！）
git push origin --force --all
```

---

## 🔧 故障排除

### 问题1: Parquet读取错误

**错误信息**:
```
ValueError: Wrong number of dimensions. values.ndim > ndim [2 > 1]
ArrowTypeError: Did not pass numpy.dtype object
```

**解决方案**:
```bash
# 1. 确保使用正确的pandas版本
pip install pandas==2.0.3

# 2. 在代码中使用fastparquet引擎
import pandas as pd
df = pd.read_parquet('file.parquet', engine='fastparquet')
```

### 问题2: 内存不足

**症状**: 
- 进程被系统杀死
- "Killed" 错误信息

**解决方案**:
1. 使用优化版本的脚本:
   ```bash
   python scripts/step1_load_data_optimized.py  # ✅ 使用这个
   # 不要用: python scripts/step1_load_data.py  # ❌
   ```

2. 减小chunk_size（在config.yaml中）:
   ```yaml
   preprocessing:
     chunk_size: 5000  # 从10000减小到5000
   ```

3. 增加系统swap空间（Linux/macOS）:
   ```bash
   # macOS会自动管理swap
   # Linux:
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### 问题3: CSV编码错误

**错误信息**:
```
ValueError: Error converting column "gsn" to bytes using encoding UTF8
```

**解决方案**:
这是prescriptions.csv的编码问题。Prescriptions是可选的，pipeline会自动跳过。

如果你需要prescriptions数据：
```python
# 尝试不同的编码
df = pd.read_csv('prescriptions.csv', encoding='latin1')
# 或
df = pd.read_csv('prescriptions.csv', encoding='iso-8859-1')
```

### 问题4: 处理速度慢

**正常速度参考**:
- Step 1: 30-40分钟
- Step 2: 5-10分钟
- Step 3: **2-3小时** ⏰（最慢，处理32万+记录）
- Step 4-6: 共10-15分钟

**优化建议**:
1. 使用SSD而不是HDD
2. 在后台运行: `nohup bash run_all.sh &`
3. Step 3是最慢的，这是正常的
4. 确保足够的RAM（推荐32GB）

### 问题5: Git推送失败

**错误**: `remote: error: File xxx.pkl is 10.00 GB; this exceeds GitHub's file size limit of 100 MB`

**解决方案**:
```bash
# 1. 检查.gitignore是否正确
cat .gitignore | grep output

# 2. 移除已添加的大文件
git rm --cached output/*.pkl
git rm --cached output/*.parquet

# 3. 提交更改
git commit -m "Remove large files"

# 4. 推送
git push
```

### 检查和监控

```bash
# 检查进度
cd preprocessing
./check_progress.sh

# 查看日志
tail -f full_pipeline.log
tail -f step3_6_full.log

# 查看输出文件
ls -lh output/

# 查看进程
ps aux | grep python | grep step

# 查看系统资源
top          # CPU和内存
df -h        # 磁盘空间
free -h      # RAM使用（Linux）
```

---

## 📈 数据统计

### 最终数据集划分

```
┌─────────────┬───────────┬────────────┬──────────────┬─────────────┐
│ 数据集      │ 样本数    │ 比例       │ 再入院数     │ 再入院率    │
├─────────────┼───────────┼────────────┼──────────────┼─────────────┤
│ 训练集      │ 194,672   │ 66.0%      │ 34,438       │ 17.69%      │
│ (2008-2013) │           │            │              │             │
├─────────────┼───────────┼────────────┼──────────────┼─────────────┤
│ 验证集      │ 55,443    │ 18.8%      │ 9,023        │ 16.27%      │
│ (2014-2016) │           │            │              │             │
├─────────────┼───────────┼────────────┼──────────────┼─────────────┤
│ 测试集      │ 44,790    │ 15.2%      │ 7,208        │ 16.09%      │
│ (2017-2019) │           │            │              │             │
├─────────────┼───────────┼────────────┼──────────────┼─────────────┤
│ 总计        │ 294,905   │ 100%       │ 50,669       │ 17.18%      │
└─────────────┴───────────┴────────────┴──────────────┴─────────────┘
```

### 年份分布

```
训练集 (2008-2013):
  ├─ 2008-2010: 128,225 样本 (65.87%)
  └─ 2011-2013: 66,447 样本 (34.13%)

验证集 (2014-2016):
  └─ 2014-2016: 55,443 样本 (100%)

测试集 (2017-2019):
  └─ 2017-2019: 44,790 样本 (100%)
```

### 队列筛选流程

```
MIMIC-IV原始数据: 546,028 住院记录
    ↓
筛选条件:
    ├─ ✅ 年龄 ≥ 18岁
    ├─ ✅ 住院时长 ≥ 48小时
    ├─ ✅ 有出院时间
    └─ ❌ 排除院内死亡
    ↓
最终队列: 322,966 住院记录 (59.2%)
    ↓
数据完整性筛选:
    └─ 有完整48小时观测数据
    ↓
最终训练数据: 294,905 样本 (91.3%)
```

### 特征统计

```
总特征数: 49

├─ 生命体征 (19个):
│  ├─ 心率 (heart_rate)
│  ├─ 收缩压 (sbp)
│  ├─ 舒张压 (dbp)
│  ├─ 平均动脉压 (map)
│  ├─ 体温 (temperature)
│  ├─ 呼吸频率 (respiratory_rate)
│  ├─ 血氧饱和度 (spo2)
│  ├─ Glasgow昏迷评分 (gcs_total, gcs_eye, gcs_verbal, gcs_motor)
│  └─ 其他生命体征...
│
└─ 实验室指标 (30个):
   ├─ 血糖 (glucose)
   ├─ 白细胞 (wbc)
   ├─ 血红蛋白 (hemoglobin)
   ├─ 血小板 (platelets)
   ├─ 肌酐 (creatinine)
   ├─ 尿素氮 (bun)
   ├─ 钠 (sodium)
   ├─ 钾 (potassium)
   ├─ 氯 (chloride)
   ├─ 碳酸氢根 (bicarbonate)
   └─ 其他实验室指标...

注: Prescriptions (药物) 特征由于数据问题未包含
```

---

## 📚 参考资料

- **MIMIC-IV文档**: https://mimic.mit.edu/docs/iv/
- **MIMIC-IV访问**: https://physionet.org/content/mimiciv/
- **论文**: (待补充)
- **GitHub仓库**: https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days

---

## 👥 贡献者

- **Yuchen Zhou** - 数据预处理Pipeline, 模型训练
- **Jiaqi** - 特征工程, 模型评估

**课程**: CS526 Machine Learning  
**机构**: Duke University  
**学期**: Fall 2025

---

## 📝 License

本项目使用的MIMIC-IV数据受限于PhysioNet的使用协议。

**数据访问要求**:
1. 完成CITI培训
2. 签署数据使用协议
3. 通过PhysioNet申请

详情: https://physionet.org/content/mimiciv/

---

## ❓ 联系方式

如有问题或建议，请：
- 提交GitHub Issue
- 或联系: [your-email]@duke.edu

---

## 🙏 致谢

感谢MIT-LCP团队维护MIMIC-IV数据集。

---

**最后更新**: 2025年11月11日
