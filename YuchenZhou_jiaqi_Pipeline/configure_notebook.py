#!/usr/bin/env python3
"""
Quick configuration helper for similarity_based_analysis.ipynb
Automatically detects available memory and recommends optimal settings.
"""

import psutil
import os

def get_memory_gb():
    """Get available memory in GB"""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)

def recommend_settings():
    """Recommend notebook settings based on available memory"""
    available_gb = get_memory_gb()
    
    print("="*70)
    print("  SIMILARITY ANALYSIS - CONFIGURATION HELPER")
    print("="*70)
    
    print(f"\n📊 System Memory:")
    mem = psutil.virtual_memory()
    print(f"  Total: {mem.total / (1024**3):.2f} GB")
    print(f"  Available: {available_gb:.2f} GB")
    print(f"  Used: {mem.percent:.1f}%")
    
    print("\n" + "="*70)
    
    if available_gb >= 12:
        print("✅ EXCELLENT: 12+ GB available")
        print("\n📝 Recommended Settings:")
        print("  USE_SUBSET = False  # Use full dataset")
        print("  FEATURE_BATCH_SIZE = 5000")
        print("  SIMILARITY_BATCH_SIZE = 1000")
        print("  BOOTSTRAP_ITERATIONS = 50")
        print("\n⏱️  Expected runtime: ~30-35 minutes")
        
    elif available_gb >= 8:
        print("✓ GOOD: 8-12 GB available")
        print("\n📝 Recommended Settings:")
        print("  USE_SUBSET = False  # Try full dataset first")
        print("  FEATURE_BATCH_SIZE = 3000  # Reduced for safety")
        print("  SIMILARITY_BATCH_SIZE = 500")
        print("  BOOTSTRAP_ITERATIONS = 50")
        print("\n💡 Tip: If notebook crashes, set USE_SUBSET = True")
        print("⏱️  Expected runtime: ~35-40 minutes")
        
    elif available_gb >= 6:
        print("⚠️  LIMITED: 6-8 GB available")
        print("\n📝 Recommended Settings:")
        print("  USE_SUBSET = True  # Use 30% of data")
        print("  SUBSET_RATIO = 0.3")
        print("  FEATURE_BATCH_SIZE = 2000")
        print("  SIMILARITY_BATCH_SIZE = 500")
        print("  BOOTSTRAP_ITERATIONS = 30")
        print("\n💡 Tips:")
        print("  - Close all unnecessary applications")
        print("  - Run during low system activity")
        print("⏱️  Expected runtime: ~10-12 minutes (subset mode)")
        
    else:
        print("❌ CRITICAL: < 6 GB available")
        print("\n⚠️  This notebook requires at least 6 GB available RAM")
        print("\n📝 Actions Required:")
        print("  1. Close all unnecessary applications")
        print("  2. Restart your computer")
        print("  3. Run immediately after restart")
        print("\n📝 Emergency Settings (if above doesn't work):")
        print("  USE_SUBSET = True")
        print("  SUBSET_RATIO = 0.2  # Use only 20% of data")
        print("  FEATURE_BATCH_SIZE = 1000")
        print("  SIMILARITY_BATCH_SIZE = 250")
        print("  BOOTSTRAP_ITERATIONS = 20")
        print("\n⏱️  Expected runtime: ~8-10 minutes (small subset)")
    
    print("\n" + "="*70)
    print("\n📖 How to apply these settings:")
    print("  1. Open similarity_based_analysis.ipynb")
    print("  2. Find the 'Memory Management Settings' cell (cell 4)")
    print("  3. Update the variables with recommended values above")
    print("  4. Run all cells sequentially")
    
    print("\n📚 For more details, see:")
    print("  - SIMILARITY_ANALYSIS_README.md")
    print("  - NOTEBOOK_OPTIMIZATION_GUIDE.md")
    
    print("\n" + "="*70)

def main():
    # Change to notebook directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    recommend_settings()
    
    # Ask if user wants to run the test
    print("\n🔧 Would you like to run full pre-flight checks?")
    print("   (This will load the training data to verify it works)")
    response = input("\nRun full checks? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n" + "="*70)
        print("  Running full pre-flight checks...")
        print("="*70)
        os.system("python3 test_notebook_requirements.py")
    else:
        print("\n✓ Configuration recommendations provided!")
        print("  Run 'python3 test_notebook_requirements.py' for full checks")

if __name__ == "__main__":
    main()
