"""
Inference module for making predictions with trained models.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any, Optional


def load_sklearn_model(model_path: str, preprocessor_path: str) -> Tuple[Any, Any]:
    """
    Load sklearn model and preprocessor.
    
    Args:
        model_path: Path to saved model (.pkl)
        preprocessor_path: Path to saved preprocessor (.joblib)
        
    Returns:
        Tuple of (model, preprocessor)
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def load_pytorch_model(model_path: str,
                      config_path: str,
                      model_type: str = 'lstm',
                      device: str = 'cpu') -> torch.nn.Module:
    """
    Load PyTorch model.
    
    Args:
        model_path: Path to saved model state dict (.pt)
        config_path: Path to config file
        model_type: 'lstm' or 'transformer'
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../training/src'))
    from models import create_lstm_model, create_transformer_model
    from utils import load_config
    
    # Load config
    config = load_config(config_path)
    
    # Load vocabulary and embedding info
    artifact_dir = os.path.dirname(model_path)
    vocab_sizes = json.load(open(os.path.join(artifact_dir, f'{model_type}_vocab_sizes.json')))
    embedding_dims = json.load(open(os.path.join(artifact_dir, f'{model_type}_embedding_dims.json')))
    
    # Get continuous dimension from saved scaler
    scaler = joblib.load(os.path.join(artifact_dir, f'{model_type}_scaler.joblib'))
    continuous_dim = scaler.n_features_in_
    
    # Create model
    if model_type == 'lstm':
        model = create_lstm_model(config, vocab_sizes, embedding_dims, continuous_dim)
    else:
        model = create_transformer_model(config, vocab_sizes, embedding_dims, continuous_dim, True)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model


def predict_sklearn(model: Any,
                   preprocessor: Any,
                   data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with sklearn model.
    
    Args:
        model: Trained sklearn model
        preprocessor: Fitted preprocessor
        data: Input dataframe
        
    Returns:
        Tuple of (predicted_labels, predicted_probabilities)
    """
    # Transform data
    X_transformed = preprocessor.transform(data)
    
    # Predict
    y_pred = model.predict(X_transformed)
    y_prob = model.predict_proba(X_transformed)[:, 1]
    
    return y_pred, y_prob


def predict_pytorch(model: torch.nn.Module,
                   data: pd.DataFrame,
                   artifact_dir: str,
                   model_type: str = 'lstm',
                   device: str = 'cpu',
                   batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with PyTorch model.
    
    Args:
        model: Trained PyTorch model
        data: Input dataframe
        artifact_dir: Directory with saved artifacts
        model_type: 'lstm' or 'transformer'
        device: Device for inference
        batch_size: Batch size
        
    Returns:
        Tuple of (predicted_labels, predicted_probabilities)
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../training/src'))
    from preprocess import transform_for_deep
    from dataset import ReadmissionDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # Load preprocessing artifacts
    with open(os.path.join(artifact_dir, f'{model_type}_cat_to_id.json'), 'r') as f:
        cat_to_id_maps = json.load(f)
    
    scaler = joblib.load(os.path.join(artifact_dir, f'{model_type}_scaler.joblib'))
    
    categorical_cols = list(cat_to_id_maps.keys())
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    
    # Remove ID and label columns from numeric
    exclude_cols = ['subject_id', 'hadm_id', 'readmit_label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and c not in categorical_cols]
    
    # Transform data
    cat_feats, cont_feats = transform_for_deep(
        data, cat_to_id_maps, scaler, categorical_cols, numeric_cols
    )
    
    # Create dummy labels (not used for prediction)
    dummy_labels = np.zeros(len(data))
    
    # Create dataset and dataloader
    dataset = ReadmissionDataset(cat_feats, cont_feats, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Predict
    all_probs = []
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                cat_batch, cont_batch, _ = batch
                mask = None
            else:
                cat_batch, cont_batch, mask, _ = batch
            
            # Move to device
            cat_batch = {k: v.to(device) for k, v in cat_batch.items()}
            cont_batch = cont_batch.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            # Forward pass
            logits = model(cat_batch, cont_batch, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_pred, y_prob


def predict(model_path: str,
           data: pd.DataFrame,
           model_type: str = 'sklearn',
           **kwargs) -> pd.DataFrame:
    """
    Universal prediction function.
    
    Args:
        model_path: Path to saved model
        data: Input dataframe
        model_type: 'sklearn', 'lstm', or 'transformer'
        **kwargs: Additional arguments (preprocessor_path, artifact_dir, etc.)
        
    Returns:
        DataFrame with predictions
    """
    if model_type == 'sklearn':
        preprocessor_path = kwargs.get('preprocessor_path')
        if not preprocessor_path:
            raise ValueError("preprocessor_path required for sklearn models")
        
        model, preprocessor = load_sklearn_model(model_path, preprocessor_path)
        y_pred, y_prob = predict_sklearn(model, preprocessor, data)
        
    elif model_type in ['lstm', 'transformer']:
        config_path = kwargs.get('config_path')
        artifact_dir = kwargs.get('artifact_dir', os.path.dirname(model_path))
        device = kwargs.get('device', 'cpu')
        batch_size = kwargs.get('batch_size', 64)
        
        model = load_pytorch_model(model_path, config_path, model_type, device)
        y_pred, y_prob = predict_pytorch(model, data, artifact_dir, model_type, device, batch_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create results dataframe
    results = pd.DataFrame({
        'predicted_label': y_pred,
        'predicted_probability': y_prob
    })
    
    # Add IDs if available
    if 'hadm_id' in data.columns:
        results.insert(0, 'hadm_id', data['hadm_id'].values)
    if 'subject_id' in data.columns:
        results.insert(0, 'subject_id', data['subject_id'].values)
    
    return results


def evaluate_predictions(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        output_dir: str = 'reports') -> Dict[str, float]:
    """
    Evaluate predictions and generate report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        output_dir: Directory to save reports
        
    Returns:
        Dictionary of metrics
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../training/src'))
    from evaluate import generate_full_report
    
    metrics = generate_full_report(
        y_true, y_pred, y_prob,
        'TEST_SET',
        output_dir
    )
    
    return metrics


def batch_predict(model_path: str,
                 input_path: str,
                 output_path: str,
                 model_type: str = 'sklearn',
                 **kwargs):
    """
    Batch prediction on a CSV file.
    
    Args:
        model_path: Path to saved model
        input_path: Path to input CSV
        output_path: Path to save predictions CSV
        model_type: 'sklearn', 'lstm', or 'transformer'
        **kwargs: Additional arguments
    """
    print(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path)
    print(f"Loaded {len(data)} samples")
    
    print(f"\nMaking predictions...")
    results = predict(model_path, data, model_type, **kwargs)
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    # Print summary
    print(f"\nPrediction summary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Predicted positive: {results['predicted_label'].sum()} ({results['predicted_label'].mean()*100:.1f}%)")
    print(f"  Mean probability: {results['predicted_probability'].mean():.3f}")
    print(f"  Probability range: [{results['predicted_probability'].min():.3f}, {results['predicted_probability'].max():.3f}]")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--model-type', type=str, default='sklearn',
                       choices=['sklearn', 'lstm', 'transformer'])
    parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor (for sklearn)')
    parser.add_argument('--config-path', type=str, help='Path to config (for pytorch)')
    parser.add_argument('--artifact-dir', type=str, help='Artifact directory (for pytorch)')
    parser.add_argument('--device', type=str, default='cpu', help='Device for pytorch models')
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.preprocessor_path:
        kwargs['preprocessor_path'] = args.preprocessor_path
    if args.config_path:
        kwargs['config_path'] = args.config_path
    if args.artifact_dir:
        kwargs['artifact_dir'] = args.artifact_dir
    if args.device:
        kwargs['device'] = args.device
    
    batch_predict(args.model_path, args.input, args.output, args.model_type, **kwargs)
