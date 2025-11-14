#!/usr/bin/env python3
"""
Quick test to verify the similarity-based analysis notebook will run successfully.
Tests memory availability and data file integrity before running the full notebook.
"""

import os
import sys
import pickle
import psutil
import numpy as np

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_memory():
    """Check available system memory"""
    print_header("MEMORY CHECK")
    
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    percent_used = mem.percent
    
    print(f"Total RAM: {total_gb:.2f} GB")
    print(f"Available RAM: {available_gb:.2f} GB")
    print(f"Used: {percent_used:.1f}%")
    
    if available_gb < 8:
        print("\n⚠️  WARNING: Less than 8GB available!")
        print("   Recommendation: Close other applications before running notebook")
        return False
    elif available_gb < 12:
        print("\n✓ Sufficient memory, but consider using batch processing")
        return True
    else:
        print("\n✓ Plenty of memory available")
        return True

def check_data_files():
    """Check if data files exist and are readable"""
    print_header("DATA FILE CHECK")
    
    base_path = "output"
    required_files = [
        "train_data.pkl",
        "val_data.pkl",
        "test_data.pkl",
        "feature_names.txt"
    ]
    
    all_good = True
    total_size = 0
    
    for filename in required_files:
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Missing: {filename}")
            all_good = False
        else:
            size_mb = os.path.getsize(filepath) / (1024**2)
            total_size += size_mb
            print(f"✓ Found: {filename} ({size_mb:.1f} MB)")
    
    print(f"\nTotal data size: {total_size/1024:.2f} GB")
    
    if not all_good:
        print("\n⚠️  ERROR: Missing required data files!")
        return False
    
    return True

def test_data_loading():
    """Test loading a small sample of data"""
    print_header("DATA LOADING TEST")
    
    try:
        print("Loading training data (this may take 1-2 minutes)...")
        with open('output/train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        
        n_samples = len(train_data['data'])
        n_features = train_data['n_features']
        n_hours = train_data['n_hours']
        
        print(f"✓ Training data loaded successfully")
        print(f"  - Samples: {n_samples:,}")
        print(f"  - Features: {n_features}")
        print(f"  - Time steps: {n_hours} hours")
        
        # Check data structure
        sample = train_data['data'][0]
        print(f"\n✓ Sample structure verified")
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - Values shape: {sample['values'].shape}")
        print(f"  - Masks shape: {sample['masks'].shape}")
        
        # Check labels
        labels = np.array([s['readmit_30d'] for s in train_data['data']])
        n_readmit = labels.sum()
        n_no_readmit = len(labels) - n_readmit
        ratio = len(labels) / n_readmit
        
        print(f"\n✓ Labels extracted")
        print(f"  - Readmitted: {n_readmit:,}")
        print(f"  - Not readmitted: {n_no_readmit:,}")
        print(f"  - Imbalance ratio: {ratio:.2f}:1")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        return False

def estimate_memory_usage():
    """Estimate memory usage for the notebook"""
    print_header("MEMORY USAGE ESTIMATION")
    
    # Rough estimates based on data size
    print("Estimated peak memory usage:")
    print("  - Data loading: ~12 GB")
    print("  - Feature extraction: ~3 GB")
    print("  - Similarity computation: ~5 GB")
    print("  - Bootstrap: ~7 GB")
    print("  - Peak usage: ~12-15 GB")
    
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb < 12:
        print(f"\n⚠️  Current available: {available_gb:.2f} GB")
        print("   Recommendation: Use subset or reduce batch sizes")
        return False
    else:
        print(f"\n✓ Current available: {available_gb:.2f} GB (sufficient)")
        return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("DEPENDENCY CHECK")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'scipy', 'tqdm', 'psutil'
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_good = False
    
    if not all_good:
        print("\n⚠️  Missing packages. Install with:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm psutil")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print("  SIMILARITY-BASED ANALYSIS NOTEBOOK")
    print("  Pre-flight Check")
    print("="*70)
    
    # Change to notebook directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Memory", check_memory),
        ("Data Files", check_data_files),
        ("Data Loading", test_data_loading),
        ("Memory Estimation", estimate_memory_usage),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou can now run the notebook:")
        print("  jupyter notebook similarity_based_analysis.ipynb")
        print("\nOr in VS Code:")
        print("  Open similarity_based_analysis.ipynb and run all cells")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease address the issues above before running the notebook.")
        print("See NOTEBOOK_OPTIMIZATION_GUIDE.md for troubleshooting tips.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
