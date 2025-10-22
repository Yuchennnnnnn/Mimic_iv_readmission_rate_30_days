"""
Feature selection module using pre-computed feature importance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def load_feature_importance(filepath: str) -> pd.DataFrame:
    """
    Load pre-computed feature importance from CSV file.
    
    Expected format:
    - feature: feature name
    - coef: coefficient value
    - importance: absolute value of coefficient
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with feature importance
    """
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['feature', 'coef', 'importance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Try alternate formats
        if 'Feature' in df.columns:
            df.rename(columns={'Feature': 'feature'}, inplace=True)
        if 'Coefficient' in df.columns:
            df.rename(columns={'Coefficient': 'coef'}, inplace=True)
        if 'Importance' in df.columns:
            df.rename(columns={'Importance': 'importance'}, inplace=True)
    
    # If importance column missing, calculate from coef
    if 'importance' not in df.columns and 'coef' in df.columns:
        df['importance'] = df['coef'].abs()
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"\nLoaded {len(df)} features from importance file")
    print(f"Top 5 features:")
    for idx, row in df.head().iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    return df


def select_top_features(feature_importance_df: pd.DataFrame,
                       top_n: Optional[int] = None,
                       importance_threshold: Optional[float] = None) -> List[str]:
    """
    Select top features based on importance.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        top_n: Number of top features to select
        importance_threshold: Minimum importance threshold
        
    Returns:
        List of selected feature names
    """
    df = feature_importance_df.copy()
    
    # Filter by threshold if specified
    if importance_threshold is not None:
        df = df[df['importance'] >= importance_threshold]
        print(f"After threshold filter (>= {importance_threshold}): {len(df)} features")
    
    # Select top N if specified
    if top_n is not None:
        df = df.head(top_n)
        print(f"Selected top {top_n} features")
    
    selected_features = df['feature'].tolist()
    
    return selected_features


def map_feature_to_column(feature_name: str, available_columns: List[str]) -> Optional[str]:
    """
    Map a feature name from importance file to actual column name in data.
    
    Handles cases like:
    - One-hot encoded features: 'gender_F' -> 'gender'
    - Exact matches: 'age' -> 'age'
    
    Args:
        feature_name: Feature name from importance file
        available_columns: List of actual column names in data
        
    Returns:
        Matched column name or None
    """
    # Direct match
    if feature_name in available_columns:
        return feature_name
    
    # Try to extract base column name from one-hot encoded feature
    # e.g., 'gender_F' -> 'gender'
    if '_' in feature_name:
        base_name = feature_name.rsplit('_', 1)[0]
        if base_name in available_columns:
            return base_name
    
    # Try lowercase/uppercase variations
    for col in available_columns:
        if col.lower() == feature_name.lower():
            return col
    
    return None


def filter_data_by_importance(data: pd.DataFrame,
                              feature_importance_path: str,
                              top_n: Optional[int] = None,
                              importance_threshold: Optional[float] = None,
                              keep_id_cols: List[str] = None,
                              keep_label_col: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter dataset to keep only important features.
    
    Args:
        data: Input dataframe
        feature_importance_path: Path to feature importance CSV
        top_n: Number of top features to keep
        importance_threshold: Minimum importance threshold
        keep_id_cols: ID columns to always keep
        keep_label_col: Label column to always keep
        
    Returns:
        Tuple of (filtered_dataframe, list_of_selected_columns)
    """
    # Load feature importance
    feat_imp_df = load_feature_importance(feature_importance_path)
    
    # Select top features
    selected_features = select_top_features(feat_imp_df, top_n, importance_threshold)
    
    # Map to actual column names
    mapped_columns = []
    unmapped_features = []
    
    for feat in selected_features:
        col = map_feature_to_column(feat, data.columns.tolist())
        if col and col not in mapped_columns:
            mapped_columns.append(col)
        elif not col:
            unmapped_features.append(feat)
    
    if unmapped_features:
        print(f"\nWarning: Could not map {len(unmapped_features)} features:")
        for feat in unmapped_features[:10]:  # Show first 10
            print(f"  - {feat}")
    
    print(f"\nSuccessfully mapped {len(mapped_columns)} features")
    
    # Add ID and label columns if specified
    cols_to_keep = mapped_columns.copy()
    
    if keep_id_cols:
        for col in keep_id_cols:
            if col in data.columns and col not in cols_to_keep:
                cols_to_keep.append(col)
    
    if keep_label_col and keep_label_col in data.columns and keep_label_col not in cols_to_keep:
        cols_to_keep.append(keep_label_col)
    
    # Filter data
    filtered_data = data[cols_to_keep].copy()
    
    print(f"\nFiltered dataset:")
    print(f"  Original: {data.shape}")
    print(f"  Filtered: {filtered_data.shape}")
    print(f"  Kept {len(mapped_columns)} feature columns")
    
    return filtered_data, mapped_columns


def get_feature_importance_for_columns(feature_importance_path: str,
                                      columns: List[str]) -> Optional[np.ndarray]:
    """
    Get importance values for specific columns.
    
    Args:
        feature_importance_path: Path to feature importance CSV
        columns: List of column names
        
    Returns:
        Array of importance values or None
    """
    feat_imp_df = load_feature_importance(feature_importance_path)
    
    # Create mapping
    importance_dict = dict(zip(feat_imp_df['feature'], feat_imp_df['importance']))
    
    # Get importance for each column
    importances = []
    for col in columns:
        # Try direct match
        if col in importance_dict:
            importances.append(importance_dict[col])
        else:
            # Try to find any one-hot encoded version
            matched = False
            for feat in importance_dict.keys():
                if feat.startswith(col + '_'):
                    importances.append(importance_dict[feat])
                    matched = True
                    break
            if not matched:
                importances.append(0.0)  # Not found, assign 0
    
    return np.array(importances)


def print_feature_importance_summary(feature_importance_path: str, top_n: int = 20):
    """
    Print summary of feature importance.
    
    Args:
        feature_importance_path: Path to feature importance CSV
        top_n: Number of top features to show
    """
    feat_imp_df = load_feature_importance(feature_importance_path)
    
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal features: {len(feat_imp_df)}")
    print(f"\nTop {top_n} most important features:")
    print(f"{'Rank':<6} {'Feature':<50} {'Coef':<12} {'Importance':<12}")
    print(f"{'-'*80}")
    
    for idx, row in feat_imp_df.head(top_n).iterrows():
        print(f"{idx+1:<6} {row['feature']:<50} {row['coef']:>11.4f} {row['importance']:>11.4f}")
    
    print(f"{'='*80}\n")
