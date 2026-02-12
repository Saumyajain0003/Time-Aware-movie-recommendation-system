#!/usr/bin/env python
"""
Quick start guide for Time-Aware Recommender System
Run this file to get started immediately
"""

import os
import sys
from pathlib import Path

def main():
    print("\n" + "=" * 70)
    print("TIME-AWARE RECOMMENDER SYSTEM - QUICK START")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("\n❌ Error: Please run this script from the project root directory")
        print("   (the directory containing 'src', 'data', 'models', etc.)")
        sys.exit(1)
    
    # Check if dependencies are installed
    print("\n1. Checking dependencies...")
    try:
        import numpy
        import pandas
        import scipy
        from sklearn import preprocessing
        print("   ✓ All dependencies are installed")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        print("\n   Install dependencies with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    print("\n2. Creating directories...")
    for dir_path in ["data", "models", "results", "notebooks"]:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"   ✓ {dir_path}/")
    
    # Ask user what to do
    print("\n" + "=" * 70)
    print("What would you like to do?")
    print("=" * 70)
    print("""
1. Run complete pipeline (generate data, train models, evaluate)
2. Run demo script (interactive demos)
3. View existing results
4. Just setup (don't run anything yet)
""")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 70)
        print("Running complete pipeline...")
        print("=" * 70)
        os.chdir("src")
        os.system("python pipeline.py")
        os.chdir("..")
        
        print("\n✓ Pipeline complete!")
        print("  Check 'results/' directory for outputs")
        
    elif choice == "2":
        print("\n" + "=" * 70)
        print("Starting demo...")
        print("=" * 70)
        os.chdir("src")
        os.system("python demo.py")
        os.chdir("..")
        
    elif choice == "3":
        print("\n" + "=" * 70)
        print("Checking for existing results...")
        print("=" * 70)
        
        results_dir = Path("results")
        if not results_dir.exists():
            print("❌ No results found. Run the pipeline first:")
            print("   cd src")
            print("   python pipeline.py")
            sys.exit(1)
        
        # Show existing files
        print("\n✓ Found results:")
        for file in results_dir.glob("*"):
            print(f"   - {file}")
        
        # Try to read and display
        json_file = results_dir / "evaluation_results.json"
        if json_file.exists():
            print("\n" + "=" * 70)
            print("Evaluation Results:")
            print("=" * 70)
            with open(json_file, 'r') as f:
                import json
                results = json.load(f)
                for model, metrics in results.items():
                    print(f"\n{model.upper()}:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
        
        summary_file = results_dir / "summary_report.txt"
        if summary_file.exists():
            print("\n" + "=" * 70)
            print("Summary Report:")
            print("=" * 70)
            with open(summary_file, 'r') as f:
                print(f.read())
    
    elif choice == "4":
        print("\n✓ Setup complete! You're ready to go.")
        print("\nNext steps:")
        print("  1. cd src")
        print("  2. python pipeline.py")
        print("\nOr run this script again and choose option 1")
    
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("For more information, see README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
