# MIMIC-IV Readmission Feature Pipeline

This project generates and cleans an **enhanced readmission feature dataset** based on the **MIMIC-IV v3.1** hospital data.

---

## 📂 Folder Setup

Before running, make sure your raw MIMIC-IV data is located under:

```
datasets/mimic-iv-3.1/hosp/
```

The folder should contain files such as:

```
patients.csv
admissions.csv
diagnoses_icd.csv
transfers.csv
services.csv
labevents.csv
d_labitems.csv
omr.csv
```

---

## 🧩 Step 1 — Feature Generation (`generate_readmission_features_step1.py`)

This script processes the raw **MIMIC-IV tables** and generates a feature dataset with:

* Demographics, admission, and discharge information
* Readmission label within 30 or 60 days
* Mortality, diagnosis complexity, service and transfer features
* OMR (BMI, eGFR) and lab test statistics (min/median/max)

**Output:**

```
datasets/readmission_features_30d_v1.csv
datasets/readmission_features_60d_v1.csv
```

You can specify the window using:

```bash
python generate_readmission_features_step1.py --window_days 30
```

---

## 🧹 Step 2 — Data Cleaning (`generate_readmission_features_step2.py`)

This script performs strict data cleaning on the generated dataset:

* Removes rows with missing key fields (subject\_id, hadm\_id, time columns, label)
* Drops columns with more than 70% missing values
* Drops any remaining rows containing NaN values
* Ensures time validity (discharge time must be after admit time)

**Output:**

```
datasets/cleaned_data.csv
```

Run:

```bash
python generate_readmission_features_step2.py
```

---

## ✅ Final Result

After completing both steps, you will have a **fully cleaned, structured dataset** ready for model training or statistical analysis:

```
datasets/cleaned_data.csv
```

---

**Pipeline Summary:**

1. Step 1 → Generate feature-rich dataset from MIMIC-IV raw tables
2. Step 2 → Strictly clean and validate the data (no interpolation or fi
