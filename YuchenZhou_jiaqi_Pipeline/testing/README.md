# Testing & Inference Module

This directory contains scripts and utilities for testing trained models on new data.

## 📁 Directory Structure

```
testing/
├── src/
│   ├── inference.py       # Load models and make predictions
│   └── batch_predict.py   # Batch prediction script
├── artifacts/             # Copied from training/ (models and encoders)
├── reports/               # Test set evaluation results
└── README.md              # This file
```

## 🚀 Usage

### 1. Copy Trained Artifacts

After training models, copy artifacts to testing directory:

```bash
cp -r ../training/artifacts/* artifacts/
```

### 2. Run Inference

**Single model prediction:**

```python
from src.inference import load_model, predict

# Load model and preprocessor
model, preprocessor = load_model('artifacts/lr.pkl', 'artifacts/lr_preprocessor.joblib')

# Make predictions
predictions = predict(model, preprocessor, new_data_df)
```

**Batch predictions:**

```bash
python src/batch_predict.py \
  --model-path artifacts/lr.pkl \
  --preprocessor-path artifacts/lr_preprocessor.joblib \
  --input data/test_data.csv \
  --output predictions.csv
```

### 3. Evaluate on Test Set

If you have labels for your test set:

```python
from src.inference import evaluate_test_set

metrics = evaluate_test_set(
    model_path='artifacts/lr.pkl',
    preprocessor_path='artifacts/lr_preprocessor.joblib',
    test_data_path='data/test_with_labels.csv',
    output_dir='reports/'
)
```

## 📊 Expected Outputs

- `predictions.csv` - Predicted probabilities and labels
- `test_metrics.csv` - Evaluation metrics (if labels available)
- `test_roc_curve.png` - ROC curve plot
- `test_confusion_matrix.png` - Confusion matrix

## 🔧 Advanced Usage

### Deploy as REST API

Use Flask or FastAPI to serve predictions:

```python
from flask import Flask, request, jsonify
from src.inference import load_model, predict

app = Flask(__name__)
model, preprocessor = load_model('artifacts/lr.pkl', 'artifacts/lr_preprocessor.joblib')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    predictions = predict(model, preprocessor, data)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Model Monitoring

Track prediction distribution over time to detect drift:

```python
from src.inference import monitor_predictions

monitor_predictions(
    predictions_dir='reports/',
    reference_dist='training/reports/predictions_lr.csv'
)
```

## 📝 Notes

- Ensure test data has the same columns as training data
- Missing columns will be filled with defaults
- New categorical values will be mapped to `<UNK>`
- For PyTorch models, ensure same PyTorch version as training
