"""
Training script for readmission prediction models.

Supports: Logistic Regression, Random Forest, XGBoost, LSTM, Transformer
"""

import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

import preprocess
import models
from dataset import ReadmissionDataset, create_dataloaders
from utils import load_config, set_seed, get_device, ensure_dir, EarlyStopping, count_parameters
from evaluate import generate_full_report
from feature_selection import filter_data_by_importance, print_feature_importance_summary


def train_sklearn_model(model, X_train, y_train, X_test, y_test, 
                       model_name, config, output_dir):
    """
    Train sklearn-based model (LR, RF, XGB).
    
    Args:
        model: Sklearn model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        config: Configuration dictionary
        output_dir: Directory to save artifacts
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Train model
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Predictions
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    reports_dir = config['data']['reports_dir']
    ensure_dir(reports_dir)
    
    # Get feature importance if available
    feature_names = None
    feature_importances = None
    
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        # Load feature names
        feature_names_path = os.path.join(output_dir, f'{model_name}_feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
    elif hasattr(model, 'coef_'):
        feature_importances = np.abs(model.coef_[0])
        # Load feature names
        feature_names_path = os.path.join(output_dir, f'{model_name}_feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
    
    metrics = generate_full_report(
        y_test, y_pred, y_prob, 
        model_name.upper(),
        reports_dir,
        feature_names=feature_names,
        feature_importances=feature_importances
    )
    
    return model, metrics


def train_pytorch_model(model, train_loader, val_loader, model_name, config, output_dir, device):
    """
    Train PyTorch-based model (LSTM, Transformer).
    
    Args:
        model: PyTorch model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model_name: Name of the model
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        device: Torch device
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Model to device
    model = model.to(device)
    
    # Training parameters
    params = config['hyperparameters'][model_name]
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    patience = params['early_stopping_patience']
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    best_auc = 0
    best_epoch = 0
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            if len(batch) == 3:  # Non-sequential
                cat_feats, cont_feats, labels = batch
                mask = None
            else:  # Sequential
                cat_feats, cont_feats, mask, labels = batch
            
            # Move to device
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
            cont_feats = cont_feats.to(device)
            if mask is not None:
                mask = mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(cat_feats, cont_feats, mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for batch in val_pbar:
                if len(batch) == 3:  # Non-sequential
                    cat_feats, cont_feats, labels = batch
                    mask = None
                else:  # Sequential
                    cat_feats, cont_feats, mask, labels = batch
                
                # Move to device
                cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                cont_feats = cont_feats.to(device)
                if mask is not None:
                    mask = mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(cat_feats, cont_feats, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Get probabilities
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Compute validation AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)
        history['val_auc'].append(val_auc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, f'{model_name}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (AUC: {best_auc:.4f})")
        
        # Early stopping
        if early_stopping(val_auc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete!")
    print(f"Best validation AUC: {best_auc:.4f} (epoch {best_epoch})")
    
    # Save training history
    history_path = os.path.join(output_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for evaluation
    model_path = os.path.join(output_dir, f'{model_name}.pt')
    model.load_state_dict(torch.load(model_path))
    
    return model, history


def evaluate_pytorch_model(model, test_loader, model_name, config, device):
    """
    Evaluate PyTorch model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        model_name: Name of the model
        config: Configuration dictionary
        device: Torch device
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} on test set")
    print(f"{'='*60}")
    
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            if len(batch) == 3:  # Non-sequential
                cat_feats, cont_feats, labels = batch
                mask = None
            else:  # Sequential
                cat_feats, cont_feats, mask, labels = batch
            
            # Move to device
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
            cont_feats = cont_feats.to(device)
            if mask is not None:
                mask = mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(cat_feats, cont_feats, mask)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy
    y_test = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Evaluate
    reports_dir = config['data']['reports_dir']
    ensure_dir(reports_dir)
    
    metrics = generate_full_report(
        y_test, y_pred, y_prob,
        model_name.upper(),
        reports_dir
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train readmission prediction models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'logistic', 'rf', 'xgb', 'lstm', 'transformer'],
                       help='Model to train')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        for model in ['lstm', 'transformer']:
            if model in config['hyperparameters']:
                config['hyperparameters'][model]['num_epochs'] = args.epochs
    
    if args.batch_size is not None:
        for model in ['lstm', 'transformer']:
            if model in config['hyperparameters']:
                config['hyperparameters'][model]['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        for model in ['lstm', 'transformer']:
            if model in config['hyperparameters']:
                config['hyperparameters'][model]['learning_rate'] = args.learning_rate
    
    # Set seed
    set_seed(config['seed'], config['deterministic'])
    
    # Get device
    device = get_device(config['device'])
    
    # Load data
    data_path = config['data']['input_path']
    print(f"\nLoading data from {data_path}...")
    df = preprocess.load_data(data_path)
    
    # Feature selection using pre-computed importance (if enabled)
    if config.get('feature_selection', {}).get('enabled', False):
        feature_imp_path = config['data'].get('feature_importance_path')
        if feature_imp_path and os.path.exists(feature_imp_path):
            print("\n" + "="*60)
            print("FEATURE SELECTION")
            print("="*60)
            print_feature_importance_summary(feature_imp_path, top_n=20)
            
            # Get parameters
            top_n = config['feature_selection'].get('top_n')
            threshold = config['feature_selection'].get('importance_threshold')
            
            # Filter data
            df_filtered, selected_features = filter_data_by_importance(
                df, 
                feature_imp_path,
                top_n=top_n,
                importance_threshold=threshold,
                keep_id_cols=config['columns']['id_cols'],
                keep_label_col=config['columns']['label']
            )
            df = df_filtered
            print(f"\nUsing {len(selected_features)} selected features for training")
        else:
            print(f"\nWarning: Feature importance file not found: {feature_imp_path}")
            print("Using all features...")
    
    # Detect column types (after feature selection)
    categorical_cols, numeric_cols = preprocess.detect_column_types(df, config)
    
    # Filter column lists to only include columns that are still in df
    available_cols = set(df.columns)
    categorical_cols = [col for col in categorical_cols if col in available_cols]
    numeric_cols = [col for col in numeric_cols if col in available_cols]
    
    print(f"\nDetected {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns after filtering")
    
    # Handle missing values
    df = preprocess.handle_missing_values(df, categorical_cols, numeric_cols, config)
    
    # Split data
    print("\nSplitting data...")
    id_cols = config['columns']['id_cols']
    label_col = config['columns']['label']
    
    X = df.drop(columns=id_cols + [label_col] + config['columns']['time_cols'], errors='ignore')
    y = df[label_col].values
    ids = df[id_cols[1]].values if len(id_cols) > 1 else df[id_cols[0]].values
    
    # Train-test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state'],
        stratify=y if config['split']['stratify'] else None
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Create output directory
    output_dir = config['data']['output_dir']
    ensure_dir(output_dir)
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = config['models_to_run']
    else:
        models_to_train = [args.model]
    
    print(f"\nModels to train: {models_to_train}")
    
    # Train models
    all_metrics = {}
    
    for model_name in models_to_train:
        try:
            if model_name == 'logistic':
                # Build preprocessor
                print("\n" + "="*60)
                print("LOGISTIC REGRESSION")
                print("="*60)
                encoder_info = preprocess.build_encoders_for_lr(
                    X_train, categorical_cols, numeric_cols, config, output_dir
                )
                
                # Transform data
                X_train_lr = preprocess.transform_for_lr(X_train, encoder_info['preprocessor'])
                X_test_lr = preprocess.transform_for_lr(X_test, encoder_info['preprocessor'])
                
                # Create and train model
                model = models.create_logistic_regression(config)
                trained_model, metrics = train_sklearn_model(
                    model, X_train_lr, y_train, X_test_lr, y_test,
                    'lr', config, output_dir
                )
                all_metrics['logistic'] = metrics
                
            elif model_name == 'rf':
                # Build preprocessor
                print("\n" + "="*60)
                print("RANDOM FOREST")
                print("="*60)
                encoder_info = preprocess.build_encoders_for_rf(
                    X_train, categorical_cols, numeric_cols, config, output_dir
                )
                
                # Transform data
                X_train_rf = preprocess.transform_for_rf(X_train, encoder_info['preprocessor'])
                X_test_rf = preprocess.transform_for_rf(X_test, encoder_info['preprocessor'])
                
                # Create and train model
                model = models.create_random_forest(config)
                trained_model, metrics = train_sklearn_model(
                    model, X_train_rf, y_train, X_test_rf, y_test,
                    'rf', config, output_dir
                )
                all_metrics['rf'] = metrics
                
            elif model_name == 'xgb':
                # Build preprocessor
                print("\n" + "="*60)
                print("XGBOOST")
                print("="*60)
                encoder_info = preprocess.build_encoders_for_xgb(
                    X_train, categorical_cols, numeric_cols, config, output_dir
                )
                
                # Transform data
                X_train_xgb = preprocess.transform_for_xgb(X_train, encoder_info['preprocessor'])
                X_test_xgb = preprocess.transform_for_xgb(X_test, encoder_info['preprocessor'])
                
                # Create and train model
                model = models.create_xgboost(config)
                trained_model, metrics = train_sklearn_model(
                    model, X_train_xgb, y_train, X_test_xgb, y_test,
                    'xgb', config, output_dir
                )
                all_metrics['xgb'] = metrics
                
            elif model_name in ['lstm', 'transformer']:
                # Build preprocessor for deep models
                print("\n" + "="*60)
                print(model_name.upper())
                print("="*60)
                encoder_info = preprocess.build_encoders_for_deep(
                    X_train, categorical_cols, numeric_cols, config, output_dir, model_name
                )
                
                # Transform data
                cat_train, cont_train = preprocess.transform_for_deep(
                    X_train, encoder_info['cat_to_id_maps'], encoder_info['scaler'],
                    encoder_info['categorical_cols'], encoder_info['numeric_cols']
                )
                cat_test, cont_test = preprocess.transform_for_deep(
                    X_test, encoder_info['cat_to_id_maps'], encoder_info['scaler'],
                    encoder_info['categorical_cols'], encoder_info['numeric_cols']
                )
                
                # Split train into train and validation
                train_indices, val_indices = train_test_split(
                    np.arange(len(y_train)),
                    test_size=0.15,
                    random_state=config['seed'],
                    stratify=y_train
                )
                
                # Create datasets
                train_dataset = ReadmissionDataset(
                    {k: v[train_indices] for k, v in cat_train.items()},
                    cont_train[train_indices],
                    y_train[train_indices]
                )
                
                val_dataset = ReadmissionDataset(
                    {k: v[val_indices] for k, v in cat_train.items()},
                    cont_train[val_indices],
                    y_train[val_indices]
                )
                
                test_dataset = ReadmissionDataset(
                    cat_test,
                    cont_test,
                    y_test
                )
                
                # Create dataloaders
                batch_size = config['hyperparameters'][model_name]['batch_size']
                train_loader, val_loader = create_dataloaders(
                    train_dataset, val_dataset, batch_size
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: ReadmissionDataset.__class__.__bases__[0].__dict__['collate_fn'](b)
                )
                
                # Import collate_fn properly
                from dataset import collate_fn
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_fn
                )
                
                # Create model
                continuous_dim = cont_train.shape[1]
                
                if model_name == 'lstm':
                    model = models.create_lstm_model(
                        config, encoder_info['vocab_sizes'],
                        encoder_info['embedding_dims'], continuous_dim
                    )
                else:  # transformer
                    model = models.create_transformer_model(
                        config, encoder_info['vocab_sizes'],
                        encoder_info['embedding_dims'], continuous_dim,
                        use_tab_transformer=True  # Use TabTransformer for single timestep
                    )
                
                # Train model
                trained_model, history = train_pytorch_model(
                    model, train_loader, val_loader, model_name, config, output_dir, device
                )
                
                # Evaluate on test set
                metrics = evaluate_pytorch_model(
                    trained_model, test_loader, model_name, config, device
                )
                all_metrics[model_name] = metrics
                
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison report
    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        metrics_df = pd.read_csv(os.path.join(config['data']['reports_dir'], 'metrics.csv'))
        print("\n", metrics_df)
        
        from evaluate import compare_models
        compare_models(
            metrics_df,
            os.path.join(config['data']['reports_dir'], 'model_comparison.png')
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Artifacts saved to: {output_dir}")
    print(f"Reports saved to: {config['data']['reports_dir']}")


if __name__ == '__main__':
    main()
