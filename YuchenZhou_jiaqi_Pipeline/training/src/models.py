"""
Model definitions for readmission prediction.

Includes:
- Sklearn-based models: Logistic Regression, Random Forest, XGBoost
- PyTorch models: LSTM, Transformer
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# ============================================================================
# Scikit-learn Models
# ============================================================================

def create_logistic_regression(config: Dict) -> LogisticRegression:
    """
    Create Logistic Regression model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LogisticRegression instance
    """
    params = config['hyperparameters']['logistic']
    model = LogisticRegression(**params)
    return model


def create_random_forest(config: Dict) -> RandomForestClassifier:
    """
    Create Random Forest model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RandomForestClassifier instance
    """
    params = config['hyperparameters']['rf']
    model = RandomForestClassifier(**params)
    return model


def create_xgboost(config: Dict) -> xgb.XGBClassifier:
    """
    Create XGBoost model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        XGBClassifier instance
    """
    params = config['hyperparameters']['xgb']
    model = xgb.XGBClassifier(**params)
    return model


# ============================================================================
# PyTorch Models
# ============================================================================

class LSTMReadmit(nn.Module):
    """
    LSTM-based model for readmission prediction.
    Supports multiple categorical features with embeddings and continuous features.
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 embedding_dims: Dict[str, int],
                 continuous_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.3):
        """
        Args:
            vocab_sizes: Dictionary mapping categorical column to vocabulary size
            embedding_dims: Dictionary mapping categorical column to embedding dimension
            continuous_dim: Number of continuous features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
        """
        super(LSTMReadmit, self).__init__()
        
        self.cat_cols = list(vocab_sizes.keys())
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Create embedding layer for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], embedding_dims[col], padding_idx=0)
            for col in self.cat_cols
        })
        
        # Calculate input dimension
        total_emb_dim = sum(embedding_dims.values())
        input_dim = total_emb_dim + continuous_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1)
    
    def forward(self,
                cat_features: Dict[str, torch.Tensor],
                cont_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cat_features: Dictionary of categorical feature tensors (B, T)
            cont_features: Continuous features (B, T, cont_dim)
            mask: Attention mask (B, T) - optional
            
        Returns:
            Logits (B,)
        """
        batch_size = cont_features.size(0)
        seq_len = cont_features.size(1)
        
        # Embed categorical features
        embeddings = []
        for col in self.cat_cols:
            emb = self.embeddings[col](cat_features[col])  # (B, T, emb_dim)
            embeddings.append(emb)
        
        # Concatenate all embeddings with continuous features
        if embeddings:
            x = torch.cat(embeddings + [cont_features], dim=-1)  # (B, T, input_dim)
        else:
            x = cont_features
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, T, hidden_dim * num_directions)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim * 2)
        else:
            h_last = h_n[-1]  # (B, hidden_dim)
        
        # Output layer
        out = self.dropout(h_last)
        logits = self.fc(out).squeeze(-1)  # (B,)
        
        return logits


class TransformerReadmit(nn.Module):
    """
    Transformer-based model for readmission prediction.
    Uses multi-head attention mechanism.
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 embedding_dims: Dict[str, int],
                 continuous_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            vocab_sizes: Dictionary mapping categorical column to vocabulary size
            embedding_dims: Dictionary mapping categorical column to embedding dimension
            continuous_dim: Number of continuous features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerReadmit, self).__init__()
        
        self.cat_cols = list(vocab_sizes.keys())
        self.d_model = d_model
        
        # Create embedding layer for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], embedding_dims[col], padding_idx=0)
            for col in self.cat_cols
        })
        
        # Calculate input dimension
        total_emb_dim = sum(embedding_dims.values())
        input_dim = total_emb_dim + continuous_dim
        
        # Project input to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # Max seq len = 100
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self,
                cat_features: Dict[str, torch.Tensor],
                cont_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cat_features: Dictionary of categorical feature tensors (B, T)
            cont_features: Continuous features (B, T, cont_dim)
            mask: Attention mask (B, T) - optional
            
        Returns:
            Logits (B,)
        """
        batch_size = cont_features.size(0)
        seq_len = cont_features.size(1)
        
        # Embed categorical features
        embeddings = []
        for col in self.cat_cols:
            emb = self.embeddings[col](cat_features[col])  # (B, T, emb_dim)
            embeddings.append(emb)
        
        # Concatenate all embeddings with continuous features
        if embeddings:
            x = torch.cat(embeddings + [cont_features], dim=-1)  # (B, T, input_dim)
        else:
            x = cont_features
        
        # Project to d_model
        x = self.input_projection(x)  # (B, T, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Create attention mask for transformer (True = masked)
        # Transformer expects mask in shape (B, T) where True means ignore
        if mask is not None:
            # mask is 1 for valid, 0 for padding
            # transformer needs True for padding
            attn_mask = (mask == 0)  # (B, T)
        else:
            attn_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (B, T, d_model)
        
        # Pool across sequence (mean pooling over valid tokens)
        if mask is not None:
            # Mask out padded positions and compute mean
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)
        
        # Output layer
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)  # (B,)
        
        return logits


class TabTransformer(nn.Module):
    """
    Simplified TabTransformer for tabular data (single timestep).
    Uses transformer to learn feature interactions.
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 embedding_dims: Dict[str, int],
                 continuous_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            vocab_sizes: Dictionary mapping categorical column to vocabulary size
            embedding_dims: Dictionary mapping categorical column to embedding dimension
            continuous_dim: Number of continuous features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TabTransformer, self).__init__()
        
        self.cat_cols = list(vocab_sizes.keys())
        self.d_model = d_model
        self.n_cat_features = len(self.cat_cols)
        
        # Create embedding layer for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_model, padding_idx=0)
            for col in self.cat_cols
        })
        
        # Column embeddings (learnable position embeddings for features)
        if self.n_cat_features > 0:
            self.column_embedding = nn.Parameter(torch.randn(1, self.n_cat_features, d_model))
        
        # Transformer encoder for categorical features
        if self.n_cat_features > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP for combining transformer output with continuous features
        mlp_input_dim = d_model * self.n_cat_features + continuous_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
    
    def forward(self,
                cat_features: Dict[str, torch.Tensor],
                cont_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cat_features: Dictionary of categorical feature tensors (B,) or (B, 1)
            cont_features: Continuous features (B, cont_dim) or (B, 1, cont_dim)
            mask: Not used in this model
            
        Returns:
            Logits (B,)
        """
        batch_size = cont_features.size(0)
        
        # Handle shape: squeeze sequence dimension if present
        if len(cont_features.shape) == 3:
            cont_features = cont_features.squeeze(1)  # (B, cont_dim)
        
        # Embed categorical features
        if self.n_cat_features > 0:
            embeddings = []
            for col in self.cat_cols:
                cat_vals = cat_features[col]
                if len(cat_vals.shape) > 1:
                    cat_vals = cat_vals.squeeze(1)  # (B,)
                emb = self.embeddings[col](cat_vals)  # (B, d_model)
                embeddings.append(emb)
            
            # Stack embeddings: (B, n_cat_features, d_model)
            x_cat = torch.stack(embeddings, dim=1)
            
            # Add column embeddings
            x_cat = x_cat + self.column_embedding
            
            # Transformer encoding
            x_cat = self.transformer_encoder(x_cat)  # (B, n_cat_features, d_model)
            
            # Flatten
            x_cat = x_cat.view(batch_size, -1)  # (B, n_cat_features * d_model)
            
            # Concatenate with continuous features
            x = torch.cat([x_cat, cont_features], dim=-1)
        else:
            x = cont_features
        
        # MLP
        logits = self.mlp(x).squeeze(-1)  # (B,)
        
        return logits


def create_lstm_model(config: Dict,
                     vocab_sizes: Dict[str, int],
                     embedding_dims: Dict[str, int],
                     continuous_dim: int) -> LSTMReadmit:
    """
    Create LSTM model from config.
    
    Args:
        config: Configuration dictionary
        vocab_sizes: Vocabulary sizes for categorical features
        embedding_dims: Embedding dimensions for categorical features
        continuous_dim: Number of continuous features
        
    Returns:
        LSTMReadmit model
    """
    params = config['hyperparameters']['lstm']
    
    model = LSTMReadmit(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        continuous_dim=continuous_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        bidirectional=params['bidirectional'],
        dropout=params['dropout']
    )
    
    return model


def create_transformer_model(config: Dict,
                            vocab_sizes: Dict[str, int],
                            embedding_dims: Dict[str, int],
                            continuous_dim: int,
                            use_tab_transformer: bool = True) -> nn.Module:
    """
    Create Transformer model from config.
    
    Args:
        config: Configuration dictionary
        vocab_sizes: Vocabulary sizes for categorical features
        embedding_dims: Embedding dimensions for categorical features
        continuous_dim: Number of continuous features
        use_tab_transformer: Whether to use TabTransformer (for single timestep)
        
    Returns:
        Transformer model
    """
    params = config['hyperparameters']['transformer']
    
    if use_tab_transformer:
        model = TabTransformer(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            continuous_dim=continuous_dim,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout']
        )
    else:
        model = TransformerReadmit(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            continuous_dim=continuous_dim,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout']
        )
    
    return model
