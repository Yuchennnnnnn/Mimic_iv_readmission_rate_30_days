"""
Utility functions for the readmission prediction pipeline.
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations don't have deterministic implementations
        # which may cause warnings


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    Get torch device.
    
    Args:
        device_name: Requested device name ('cuda' or 'cpu')
        
    Returns:
        torch.device object
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy/auc
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.compare = lambda x, y: x < y - min_delta
        else:
            self.compare = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_info(model_name: str, info: Dict[str, Any], output_dir: str):
    """
    Save model information to a text file.
    
    Args:
        model_name: Name of the model
        info: Dictionary with model information
        output_dir: Directory to save the file
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f'{model_name}_info.txt')
    
    with open(filepath, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved model info to {filepath}")
