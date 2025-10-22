"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import preprocess


def create_sample_data(n_samples=100):
    """Create sample dataset for testing."""
    np.random.seed(42)
    
    data = {
        'subject_id': np.arange(n_samples),
        'hadm_id': np.arange(n_samples) + 1000,
        'readmit_label': np.random.randint(0, 2, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private'], n_samples),
        'age': np.random.randint(18, 90, n_samples),
        'los': np.random.uniform(1, 30, n_samples),
        'num_diagnoses': np.random.randint(1, 20, n_samples)
    }
    
    return pd.DataFrame(data)


def test_load_data():
    """Test data loading."""
    df = create_sample_data()
    
    # Save to temp CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    
    try:
        # Load data
        loaded_df = preprocess.load_data(temp_path, validate=False)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
    finally:
        os.unlink(temp_path)


def test_validate_data():
    """Test data validation."""
    df = create_sample_data()
    
    # Should not raise error
    preprocess.validate_data(df)
    
    # Should raise error for missing columns
    df_missing = df.drop('readmit_label', axis=1)
    with pytest.raises(ValueError):
        preprocess.validate_data(df_missing)


def test_detect_column_types():
    """Test column type detection."""
    df = create_sample_data()
    
    config = {
        'columns': {
            'id_cols': ['subject_id', 'hadm_id'],
            'time_cols': [],
            'label': 'readmit_label',
            'categorical_cols': ['gender', 'insurance'],
            'numeric_cols': ['age', 'los', 'num_diagnoses']
        }
    }
    
    cat_cols, num_cols = preprocess.detect_column_types(df, config)
    
    assert 'gender' in cat_cols
    assert 'insurance' in cat_cols
    assert 'age' in num_cols
    assert 'los' in num_cols


def test_handle_missing_values():
    """Test missing value handling."""
    df = create_sample_data()
    
    # Add some missing values
    df.loc[0, 'age'] = np.nan
    df.loc[1, 'gender'] = np.nan
    
    config = {
        'preprocessing': {
            'fill_numeric_na': 'median',
            'fill_categorical_na': 'mode'
        }
    }
    
    cat_cols = ['gender', 'insurance']
    num_cols = ['age', 'los', 'num_diagnoses']
    
    df_clean = preprocess.handle_missing_values(df, cat_cols, num_cols, config)
    
    # Check no NaNs remain
    assert not df_clean['age'].isna().any()
    assert not df_clean['gender'].isna().any()


def test_calculate_embedding_dim():
    """Test embedding dimension calculation."""
    config = {
        'embeddings': {
            'max_emb_dim': 50,
            'multiplier': 1.6,
            'exponent': 0.56
        }
    }
    
    # Small vocab
    emb_dim = preprocess.calculate_embedding_dim(10, config)
    assert emb_dim > 0 and emb_dim <= 50
    
    # Large vocab
    emb_dim = preprocess.calculate_embedding_dim(1000, config)
    assert emb_dim > 0 and emb_dim <= 50


def test_build_encoders_roundtrip():
    """Test encoding and decoding roundtrip."""
    df = create_sample_data(50)
    
    config = {
        'preprocessing': {
            'ohe_cardinality_threshold': 10,
            'rare_category_threshold': 5
        },
        'embeddings': {
            'max_emb_dim': 50,
            'multiplier': 1.6,
            'exponent': 0.56
        }
    }
    
    cat_cols = ['gender', 'insurance']
    num_cols = ['age', 'los', 'num_diagnoses']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build encoders for LR
        encoder_info = preprocess.build_encoders_for_lr(
            df, cat_cols, num_cols, config, tmpdir
        )
        
        # Transform
        X_transformed = preprocess.transform_for_lr(df, encoder_info['preprocessor'])
        
        # Check output shape
        assert X_transformed.shape[0] == len(df)
        assert X_transformed.shape[1] > 0


def test_integration_small_dataset():
    """Integration test with small synthetic dataset."""
    # Create synthetic dataset
    n_samples = 200
    df = create_sample_data(n_samples)
    
    config = {
        'columns': {
            'id_cols': ['subject_id', 'hadm_id'],
            'time_cols': [],
            'label': 'readmit_label',
            'categorical_cols': ['gender', 'insurance'],
            'numeric_cols': ['age', 'los', 'num_diagnoses']
        },
        'preprocessing': {
            'ohe_cardinality_threshold': 10,
            'rare_category_threshold': 5,
            'fill_numeric_na': 'median',
            'fill_categorical_na': 'mode'
        },
        'embeddings': {
            'max_emb_dim': 50,
            'multiplier': 1.6,
            'exponent': 0.56
        }
    }
    
    cat_cols = config['columns']['categorical_cols']
    num_cols = config['columns']['numeric_cols']
    
    # Handle missing values
    df_clean = preprocess.handle_missing_values(df, cat_cols, num_cols, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test LR preprocessing
        encoder_info_lr = preprocess.build_encoders_for_lr(
            df_clean, cat_cols, num_cols, config, tmpdir
        )
        X_lr = preprocess.transform_for_lr(df_clean, encoder_info_lr['preprocessor'])
        assert X_lr.shape[0] == n_samples
        
        # Test RF preprocessing
        encoder_info_rf = preprocess.build_encoders_for_rf(
            df_clean, cat_cols, num_cols, config, tmpdir
        )
        X_rf = preprocess.transform_for_rf(df_clean, encoder_info_rf['preprocessor'])
        assert X_rf.shape[0] == n_samples
        
        # Test deep learning preprocessing
        encoder_info_deep = preprocess.build_encoders_for_deep(
            df_clean, cat_cols, num_cols, config, tmpdir, 'lstm'
        )
        cat_feats, cont_feats = preprocess.transform_for_deep(
            df_clean,
            encoder_info_deep['cat_to_id_maps'],
            encoder_info_deep['scaler'],
            encoder_info_deep['categorical_cols'],
            encoder_info_deep['numeric_cols']
        )
        assert len(cat_feats) == len(cat_cols)
        assert cont_feats.shape[0] == n_samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
