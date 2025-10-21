"""
Data preprocessing module for MIMIC-IV readmission prediction.

This module handles:
- Data loading and validation
- Encoder creation and fitting for different model types
- Feature transformation pipelines
- Artifact saving and loading
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV or parquet file with validation.
    
    Args:
        data_path: Path to the data file
        validate: Whether to validate required columns
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {data_path}...")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if validate:
        validate_data(df)
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in the dataframe.
    
    Args:
        df: Input dataframe to validate
    """
    # Check for basic required columns
    required_cols = ['subject_id', 'hadm_id', 'readmit_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check label distribution
    if 'readmit_label' in df.columns:
        label_dist = df['readmit_label'].value_counts()
        print(f"\nLabel distribution:")
        print(label_dist)
        print(f"Readmission rate: {label_dist.get(1, 0) / len(df) * 100:.2f}%")


def detect_column_types(df: pd.DataFrame, config: Dict) -> Tuple[List[str], List[str]]:
    """
    Automatically detect categorical and numeric columns.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (categorical_cols, numeric_cols)
    """
    # Get columns to exclude
    exclude_cols = (
        config['columns']['id_cols'] + 
        config['columns']['time_cols'] + 
        [config['columns']['label']]
    )
    
    # Get specified columns from config
    categorical_cols = config['columns']['categorical_cols']
    numeric_cols_config = config['columns']['numeric_cols']
    
    # Auto-detect numeric columns if not fully specified
    all_numeric = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            all_numeric.append(col)
    
    # Combine specified and detected numeric columns
    numeric_cols = list(set(numeric_cols_config + all_numeric))
    
    print(f"\nDetected {len(categorical_cols)} categorical columns")
    print(f"Detected {len(numeric_cols)} numeric columns")
    
    return categorical_cols, numeric_cols


def handle_missing_values(df: pd.DataFrame, 
                         categorical_cols: List[str],
                         numeric_cols: List[str],
                         config: Dict) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        config: Configuration dictionary
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Handle numeric columns
    fill_method = config['preprocessing'].get('fill_numeric_na', 'median')
    for col in numeric_cols:
        if df[col].isna().any():
            if fill_method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif fill_method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)
    
    # Handle categorical columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'UNKNOWN', inplace=True)
    
    return df


def calculate_embedding_dim(vocab_size: int, config: Dict) -> int:
    """
    Calculate embedding dimension using heuristic formula.
    
    Args:
        vocab_size: Size of vocabulary
        config: Configuration dictionary
        
    Returns:
        Embedding dimension
    """
    emb_config = config['embeddings']
    max_dim = emb_config['max_emb_dim']
    multiplier = emb_config['multiplier']
    exponent = emb_config['exponent']
    
    emb_dim = min(max_dim, int(multiplier * (vocab_size ** exponent)))
    return max(emb_dim, 1)  # At least 1


def build_encoders_for_lr(train_df: pd.DataFrame,
                          categorical_cols: List[str],
                          numeric_cols: List[str],
                          config: Dict,
                          output_dir: str) -> Dict[str, Any]:
    """
    Build preprocessing pipeline for Logistic Regression.
    Uses OneHotEncoder for low-cardinality categoricals, StandardScaler for numerics.
    
    Args:
        train_df: Training dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary with encoder information
    """
    threshold = config['preprocessing']['ohe_cardinality_threshold']
    
    # Separate low and high cardinality categorical columns
    ohe_cols = []
    high_card_cols = []
    
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        n_unique = train_df[col].nunique()
        if n_unique <= threshold:
            ohe_cols.append(col)
        else:
            high_card_cols.append(col)
    
    print(f"\n[LR] OHE columns ({len(ohe_cols)}): {ohe_cols}")
    print(f"[LR] High-cardinality columns (dropped): {high_card_cols}")
    
    # Build column transformer
    transformers = []
    
    if ohe_cols:
        transformers.append((
            'ohe',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ohe_cols
        ))
    
    if numeric_cols:
        transformers.append((
            'scaler',
            StandardScaler(),
            numeric_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_dir, 'lr_preprocessor.joblib'))
    
    # Save column names for reconstruction
    feature_names = []
    if ohe_cols:
        ohe = preprocessor.named_transformers_['ohe']
        feature_names.extend(ohe.get_feature_names_out(ohe_cols))
    if numeric_cols:
        feature_names.extend(numeric_cols)
    
    with open(os.path.join(output_dir, 'lr_feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    return {
        'preprocessor': preprocessor,
        'ohe_cols': ohe_cols,
        'numeric_cols': numeric_cols,
        'feature_names': feature_names
    }


def build_encoders_for_rf(train_df: pd.DataFrame,
                         categorical_cols: List[str],
                         numeric_cols: List[str],
                         config: Dict,
                         output_dir: str) -> Dict[str, Any]:
    """
    Build preprocessing pipeline for Random Forest.
    Uses OrdinalEncoder for categoricals, no scaling for numerics.
    
    Args:
        train_df: Training dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary with encoder information
    """
    # Use ordinal encoding for all categorical columns
    valid_cat_cols = [col for col in categorical_cols if col in train_df.columns]
    valid_num_cols = [col for col in numeric_cols if col in train_df.columns]
    
    transformers = []
    
    if valid_cat_cols:
        transformers.append((
            'ordinal',
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            valid_cat_cols
        ))
    
    if valid_num_cols:
        transformers.append((
            'passthrough',
            'passthrough',
            valid_num_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_dir, 'rf_preprocessor.joblib'))
    
    feature_names = valid_cat_cols + valid_num_cols
    with open(os.path.join(output_dir, 'rf_feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    return {
        'preprocessor': preprocessor,
        'categorical_cols': valid_cat_cols,
        'numeric_cols': valid_num_cols,
        'feature_names': feature_names
    }


def build_encoders_for_xgb(train_df: pd.DataFrame,
                          categorical_cols: List[str],
                          numeric_cols: List[str],
                          config: Dict,
                          output_dir: str) -> Dict[str, Any]:
    """
    Build preprocessing pipeline for XGBoost.
    Uses OrdinalEncoder for categoricals, passthrough for numerics.
    
    Args:
        train_df: Training dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary with encoder information
    """
    # Similar to RF but can use native categorical support if enabled
    valid_cat_cols = [col for col in categorical_cols if col in train_df.columns]
    valid_num_cols = [col for col in numeric_cols if col in train_df.columns]
    
    transformers = []
    
    if valid_cat_cols:
        transformers.append((
            'ordinal',
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            valid_cat_cols
        ))
    
    if valid_num_cols:
        transformers.append((
            'passthrough',
            'passthrough',
            valid_num_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_dir, 'xgb_preprocessor.joblib'))
    
    feature_names = valid_cat_cols + valid_num_cols
    with open(os.path.join(output_dir, 'xgb_feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    return {
        'preprocessor': preprocessor,
        'categorical_cols': valid_cat_cols,
        'numeric_cols': valid_num_cols,
        'feature_names': feature_names
    }


def build_encoders_for_deep(train_df: pd.DataFrame,
                           categorical_cols: List[str],
                           numeric_cols: List[str],
                           config: Dict,
                           output_dir: str,
                           model_name: str = 'lstm') -> Dict[str, Any]:
    """
    Build preprocessing for deep learning models (LSTM/Transformer).
    Creates label encoders for categoricals (starting from 1, 0 reserved for padding).
    StandardScaler for numerics.
    
    Args:
        train_df: Training dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        model_name: Name of the model ('lstm' or 'transformer')
        
    Returns:
        Dictionary with encoder information
    """
    valid_cat_cols = [col for col in categorical_cols if col in train_df.columns]
    valid_num_cols = [col for col in numeric_cols if col in train_df.columns]
    
    # Create label encoders for each categorical column
    # Reserve 0 for padding, start actual labels from 1
    label_encoders = {}
    vocab_sizes = {}
    embedding_dims = {}
    cat_to_id_maps = {}
    
    for col in valid_cat_cols:
        # Get unique values
        unique_vals = train_df[col].unique()
        unique_vals = [v for v in unique_vals if pd.notna(v)]
        
        # Create mapping: 0=PAD, 1=UNK, 2+=actual categories
        cat_to_id = {'<PAD>': 0, '<UNK>': 1}
        for idx, val in enumerate(sorted(unique_vals), start=2):
            cat_to_id[str(val)] = idx
        
        vocab_size = len(cat_to_id)
        emb_dim = calculate_embedding_dim(vocab_size, config)
        
        cat_to_id_maps[col] = cat_to_id
        vocab_sizes[col] = vocab_size
        embedding_dims[col] = emb_dim
        
        print(f"[{model_name.upper()}] {col}: vocab_size={vocab_size}, emb_dim={emb_dim}")
    
    # Create scaler for numeric columns
    scaler = StandardScaler() if valid_num_cols else None
    if scaler:
        scaler.fit(train_df[valid_num_cols])
    
    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'{model_name}_cat_to_id.json'), 'w') as f:
        json.dump(cat_to_id_maps, f, indent=2)
    
    with open(os.path.join(output_dir, f'{model_name}_vocab_sizes.json'), 'w') as f:
        json.dump(vocab_sizes, f, indent=2)
    
    with open(os.path.join(output_dir, f'{model_name}_embedding_dims.json'), 'w') as f:
        json.dump(embedding_dims, f, indent=2)
    
    if scaler:
        joblib.dump(scaler, os.path.join(output_dir, f'{model_name}_scaler.joblib'))
    
    return {
        'cat_to_id_maps': cat_to_id_maps,
        'vocab_sizes': vocab_sizes,
        'embedding_dims': embedding_dims,
        'scaler': scaler,
        'categorical_cols': valid_cat_cols,
        'numeric_cols': valid_num_cols
    }


def transform_for_lr(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Transform data using LR preprocessor.
    
    Args:
        df: Input dataframe
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        Transformed feature matrix
    """
    return preprocessor.transform(df)


def transform_for_rf(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Transform data using RF preprocessor.
    
    Args:
        df: Input dataframe
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        Transformed feature matrix
    """
    return preprocessor.transform(df)


def transform_for_xgb(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Transform data using XGB preprocessor.
    
    Args:
        df: Input dataframe
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        Transformed feature matrix
    """
    return preprocessor.transform(df)


def transform_for_deep(df: pd.DataFrame,
                      cat_to_id_maps: Dict[str, Dict],
                      scaler: Optional[StandardScaler],
                      categorical_cols: List[str],
                      numeric_cols: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Transform data for deep learning models.
    
    Args:
        df: Input dataframe
        cat_to_id_maps: Mapping dictionaries for categorical columns
        scaler: StandardScaler for numeric columns
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        
    Returns:
        Tuple of (categorical_features_dict, numeric_features_array)
    """
    # Transform categorical columns
    cat_features = {}
    for col in categorical_cols:
        cat_to_id = cat_to_id_maps[col]
        # Map values, use UNK (1) for unknown values
        cat_features[col] = df[col].apply(
            lambda x: cat_to_id.get(str(x), cat_to_id['<UNK>'])
        ).values
    
    # Transform numeric columns
    if numeric_cols and scaler:
        num_features = scaler.transform(df[numeric_cols])
    else:
        num_features = np.array([]).reshape(len(df), 0)
    
    return cat_features, num_features


def load_preprocessor(output_dir: str, model_name: str) -> Any:
    """
    Load saved preprocessor for a specific model.
    
    Args:
        output_dir: Directory containing artifacts
        model_name: Name of the model
        
    Returns:
        Loaded preprocessor or dictionary of artifacts
    """
    if model_name in ['lr', 'rf', 'xgb']:
        return joblib.load(os.path.join(output_dir, f'{model_name}_preprocessor.joblib'))
    else:
        # For deep models, load all artifacts
        with open(os.path.join(output_dir, f'{model_name}_cat_to_id.json'), 'r') as f:
            cat_to_id_maps = json.load(f)
        
        with open(os.path.join(output_dir, f'{model_name}_vocab_sizes.json'), 'r') as f:
            vocab_sizes = json.load(f)
        
        with open(os.path.join(output_dir, f'{model_name}_embedding_dims.json'), 'r') as f:
            embedding_dims = json.load(f)
        
        scaler_path = os.path.join(output_dir, f'{model_name}_scaler.joblib')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        return {
            'cat_to_id_maps': cat_to_id_maps,
            'vocab_sizes': vocab_sizes,
            'embedding_dims': embedding_dims,
            'scaler': scaler
        }
