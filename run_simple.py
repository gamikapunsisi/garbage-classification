import os
import argparse

def show_instructions():
    """Show instructions"""
    print("\n" + "="*60)
    print("GARBAGE CLASSIFICATION PROJECT")
    print("="*60)
    print("1. Setup: python minimal_setup.py")
    print("2. Download dataset from: https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
    print("3. Save as: data/raw/garbage-classification.zip")
    print("4. Process: python simple_extract.py")
    print("5. Train: python simple_train.py (after packages are installed)")
    print("="*60)

def check_setup():
    """Check if setup is complete"""
    required_dirs = [
        'data/raw',
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        'models',
        'utils'
    ]
    
    all_exists = True
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Missing: {directory}")
            all_exists = False
    
    return all_exists

def main():
    parser = argparse.ArgumentParser(description='Simple Garbage Classification Runner')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--check', action='store_true', help='Check setup status')
    parser.add_argument('--instructions', action='store_true', help='Show all instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        print("Run: python minimal_setup.py")
        
    if args.check:
        if check_setup():
            print("✓ Setup is complete")
        else:
            print("✗ Setup is incomplete")
            
    if args.instructions:
        show_instructions()
        
    if not any(vars(args).values()):
        print("Usage: python run_simple.py [--setup] [--check] [--instructions]")

if __name__ == "__main__":
    main()