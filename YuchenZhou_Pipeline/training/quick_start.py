#!/usr/bin/env python3
"""
Quick start script to test the pipeline with synthetic data.
"""

import os
import sys
import subprocess

def main():
    print("="*60)
    print("READMISSION PREDICTION PIPELINE - QUICK START")
    print("="*60)
    
    # Check if we're in the training directory
    if not os.path.exists('src/train.py'):
        print("\nError: Please run this script from the training/ directory")
        sys.exit(1)
    
    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic test data...")
    try:
        from tests.test_integration import create_synthetic_dataset
        df = create_synthetic_dataset(500)  # Small dataset for quick testing
        
        os.makedirs('data', exist_ok=True)
        output_path = 'data/synthetic_data.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Saved synthetic data to {output_path}")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        sys.exit(1)
    
    # Step 2: Update config
    print("\n[2/4] Checking configuration...")
    if os.path.exists('config.yaml'):
        print("✓ config.yaml found")
        # Could update config programmatically here if needed
    else:
        print("✗ config.yaml not found")
        sys.exit(1)
    
    # Step 3: Test with Logistic Regression (fastest)
    print("\n[3/4] Training Logistic Regression model...")
    print("-" * 60)
    
    # Update config to use synthetic data
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['input_path'] = 'data/synthetic_data.csv'
    
    with open('config_test.yaml', 'w') as f:
        yaml.dump(config, f)
    
    try:
        result = subprocess.run(
            ['python', 'src/train.py', '--model', 'logistic', '--config', 'config_test.yaml'],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ Training completed successfully!")
        else:
            print("\n✗ Training failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        sys.exit(1)
    
    # Step 4: Check outputs
    print("\n[4/4] Checking outputs...")
    
    if os.path.exists('artifacts/lr.pkl'):
        print("✓ Model saved: artifacts/lr.pkl")
    else:
        print("✗ Model file not found")
    
    if os.path.exists('reports/metrics.csv'):
        print("✓ Metrics saved: reports/metrics.csv")
        
        # Display metrics
        import pandas as pd
        metrics = pd.read_csv('reports/metrics.csv')
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(metrics.to_string(index=False))
    else:
        print("✗ Metrics file not found")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check the reports/ directory for plots")
    print("  2. Try other models:")
    print("     python src/train.py --model rf --config config_test.yaml")
    print("     python src/train.py --model xgb --config config_test.yaml")
    print("  3. Use your real data by updating config.yaml")
    print("="*60)

if __name__ == '__main__':
    main()
