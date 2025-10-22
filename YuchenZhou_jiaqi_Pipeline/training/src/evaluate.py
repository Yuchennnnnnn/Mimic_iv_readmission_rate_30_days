"""
Evaluation module for computing metrics and generating reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve


def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_prob: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # AUC scores
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['roc_auc'] = np.nan
    
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except:
        metrics['pr_auc'] = np.nan
    
    # Classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*60}")
    
    print(f"\nAUC Scores:")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1-Score:       {metrics['f1']:.4f}")
    print(f"  Specificity:    {metrics['specificity']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:>6}  FP: {metrics['false_positives']:>6}")
    print(f"  FN: {metrics['false_negatives']:>6}  TP: {metrics['true_positives']:>6}")
    print(f"{'='*60}\n")


def save_metrics(metrics: Dict[str, float],
                model_name: str,
                output_path: str):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        output_path: Path to save metrics CSV
    """
    # Add model name to metrics
    metrics_with_name = {'model': model_name, **metrics}
    
    # Convert to DataFrame
    df = pd.DataFrame([metrics_with_name])
    
    # Append to existing file or create new
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def save_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    ids: Optional[np.ndarray],
                    model_name: str,
                    output_path: str):
    """
    Save predictions to CSV file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        ids: Sample IDs (optional)
        model_name: Name of the model
        output_path: Path to save predictions CSV
    """
    data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    if ids is not None:
        data['id'] = ids
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def plot_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  model_name: str,
                  output_path: str):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def plot_pr_curve(y_true: np.ndarray,
                 y_prob: np.ndarray,
                 model_name: str,
                 output_path: str):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline (prevalence)
    baseline = y_true.mean()
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
             label=f'Baseline (prevalence = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {output_path}")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         model_name: str,
                         output_path: str):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Readmit', 'Readmit'],
                yticklabels=['No Readmit', 'Readmit'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_calibration_curve(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          model_name: str,
                          output_path: str,
                          n_bins: int = 10):
    """
    Plot and save calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save plot
        n_bins: Number of bins for calibration
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-',
             label=model_name, color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration curve to {output_path}")


def plot_feature_importance(feature_names: list,
                           importances: np.ndarray,
                           model_name: str,
                           output_path: str,
                           top_n: int = 20):
    """
    Plot and save feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        model_name: Name of the model
        output_path: Path to save plot
        top_n: Number of top features to show
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances[::-1], color='steelblue')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to {output_path}")


def generate_full_report(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        model_name: str,
                        output_dir: str,
                        ids: Optional[np.ndarray] = None,
                        feature_names: Optional[list] = None,
                        feature_importances: Optional[np.ndarray] = None):
    """
    Generate comprehensive evaluation report with plots and metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save outputs
        ids: Sample IDs (optional)
        feature_names: Feature names (optional, for importance plot)
        feature_importances: Feature importance values (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, model_name)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    save_metrics(metrics, model_name, metrics_path)
    
    # Save predictions
    preds_path = os.path.join(output_dir, f'predictions_{model_name.lower()}.csv')
    save_predictions(y_true, y_pred, y_prob, ids, model_name, preds_path)
    
    # Generate plots
    plot_roc_curve(y_true, y_prob, model_name,
                  os.path.join(output_dir, f'roc_curve_{model_name.lower()}.png'))
    
    plot_pr_curve(y_true, y_prob, model_name,
                 os.path.join(output_dir, f'pr_curve_{model_name.lower()}.png'))
    
    plot_confusion_matrix(y_true, y_pred, model_name,
                         os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png'))
    
    plot_calibration_curve(y_true, y_prob, model_name,
                          os.path.join(output_dir, f'calibration_curve_{model_name.lower()}.png'))
    
    # Feature importance (if available)
    if feature_names is not None and feature_importances is not None:
        plot_feature_importance(feature_names, feature_importances, model_name,
                              os.path.join(output_dir, f'feature_importance_{model_name.lower()}.png'))
    
    print(f"\nFull evaluation report saved to {output_dir}")
    
    return metrics


def compare_models(metrics_df: pd.DataFrame, output_path: str):
    """
    Create comparison plot for multiple models.
    
    Args:
        metrics_df: DataFrame with metrics for all models
        output_path: Path to save comparison plot
    """
    # Select key metrics for comparison
    key_metrics = ['roc_auc', 'pr_auc', 'f1', 'accuracy', 'precision', 'recall']
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if len(available_metrics) == 0:
        print("No metrics available for comparison")
        return
    
    # Create subplot for each metric
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        models = metrics_df['model'].values
        values = metrics_df[metric].values
        
        ax.bar(range(len(models)), values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to {output_path}")
