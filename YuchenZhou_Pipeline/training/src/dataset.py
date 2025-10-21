"""
PyTorch Dataset classes for LSTM and Transformer models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class ReadmissionDataset(Dataset):
    """
    Dataset for readmission prediction with categorical and continuous features.
    Supports both single-timestep and sequential data.
    """
    
    def __init__(self,
                 cat_features: Dict[str, np.ndarray],
                 cont_features: np.ndarray,
                 labels: np.ndarray,
                 max_len: Optional[int] = None):
        """
        Args:
            cat_features: Dictionary mapping column name to integer array
            cont_features: Numeric features array (n_samples, n_features)
            labels: Binary labels (n_samples,)
            max_len: Maximum sequence length (for padding)
        """
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.labels = labels
        self.max_len = max_len
        self.n_samples = len(labels)
        
        # For single timestep data, reshape to (n_samples, 1, n_features)
        if len(self.cont_features.shape) == 2:
            self.cont_features = self.cont_features[:, np.newaxis, :]
        
        # Ensure all categorical features have the same shape
        for key in self.cat_features:
            if len(self.cat_features[key].shape) == 1:
                self.cat_features[key] = self.cat_features[key][:, np.newaxis]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (cat_features_dict, cont_features, label)
        """
        # Get categorical features for this sample
        cat_dict = {}
        for key, values in self.cat_features.items():
            cat_dict[key] = torch.LongTensor(values[idx])
        
        # Get continuous features
        cont = torch.FloatTensor(self.cont_features[idx])
        
        # Get label
        label = torch.FloatTensor([self.labels[idx]])
        
        return cat_dict, cont, label


class SequenceReadmissionDataset(Dataset):
    """
    Dataset for sequential readmission prediction with variable-length sequences.
    Includes padding and attention masks.
    """
    
    def __init__(self,
                 sequences: List[Dict[str, np.ndarray]],
                 labels: np.ndarray,
                 max_len: Optional[int] = None):
        """
        Args:
            sequences: List of dictionaries, each containing 'categorical' and 'continuous' keys
            labels: Binary labels
            max_len: Maximum sequence length for padding
        """
        self.sequences = sequences
        self.labels = labels
        self.n_samples = len(labels)
        
        # Determine max length if not provided
        if max_len is None:
            self.max_len = max(len(seq['continuous']) for seq in sequences)
        else:
            self.max_len = max_len
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], 
                                             torch.Tensor, 
                                             torch.Tensor,
                                             torch.Tensor]:
        """
        Get a single sequence.
        
        Returns:
            Tuple of (cat_features_dict, cont_features, attention_mask, label)
        """
        seq = self.sequences[idx]
        seq_len = len(seq['continuous'])
        
        # Pad or truncate categorical features
        cat_dict = {}
        for key, values in seq['categorical'].items():
            if len(values) > self.max_len:
                cat_dict[key] = torch.LongTensor(values[:self.max_len])
            else:
                padded = np.zeros(self.max_len, dtype=np.int64)
                padded[:len(values)] = values
                cat_dict[key] = torch.LongTensor(padded)
        
        # Pad or truncate continuous features
        cont = seq['continuous']
        if len(cont) > self.max_len:
            cont = cont[:self.max_len]
        else:
            pad_len = self.max_len - len(cont)
            cont = np.vstack([cont, np.zeros((pad_len, cont.shape[1]))])
        cont = torch.FloatTensor(cont)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.zeros(self.max_len)
        mask[:min(seq_len, self.max_len)] = 1
        
        # Label
        label = torch.FloatTensor([self.labels[idx]])
        
        return cat_dict, cont, mask, label


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for DataLoader.
    Handles dictionary of categorical features.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    if len(batch[0]) == 3:  # Non-sequential dataset
        cat_dicts, conts, labels = zip(*batch)
        
        # Stack categorical features
        cat_batch = {}
        for key in cat_dicts[0].keys():
            cat_batch[key] = torch.stack([d[key] for d in cat_dicts])
        
        # Stack continuous and labels
        cont_batch = torch.stack(conts)
        label_batch = torch.cat(labels)
        
        return cat_batch, cont_batch, label_batch
    
    else:  # Sequential dataset with masks
        cat_dicts, conts, masks, labels = zip(*batch)
        
        # Stack categorical features
        cat_batch = {}
        for key in cat_dicts[0].keys():
            cat_batch[key] = torch.stack([d[key] for d in cat_dicts])
        
        # Stack continuous, masks, and labels
        cont_batch = torch.stack(conts)
        mask_batch = torch.stack(masks)
        label_batch = torch.cat(labels)
        
        return cat_batch, cont_batch, mask_batch, label_batch


def create_dataloaders(train_dataset: Dataset,
                      val_dataset: Dataset,
                      batch_size: int = 64,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
