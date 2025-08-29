import os
import sys

def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    return sys.version_info >= (3, 7)

def check_pip():
    """Check if pip is available"""
    try:
        import pip
        print("✓ pip is available")
        return True
    except ImportError:
        print("✗ pip is not available")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Try to install packages
    packages = [
        'tensorflow',
        'opencv-python',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pillow'
    ]
    
    for package in packages:
        try:
            print(f"Checking {package}...")
            __import__(package.lower().replace('-', '_'))
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} installed successfully")
            except Exception as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
    
    return True

def setup_directories():
    """Create necessary directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/processed/train',
        'data/processed/val', 
        'data/processed/test',
        'models',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create class subdirectories
    classes = ['plastic', 'organic', 'metal']
    for class_name in classes:
        for split in ['train', 'val', 'test']:
            path = f'data/processed/{split}/{class_name}'
            os.makedirs(path, exist_ok=True)
    
    print("Directory structure created successfully!")
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("GARBAGE CLASSIFICATION PROJECT SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("Please use Python 3.7 or higher")
        return
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        return
    
    # Install requirements
    if not install_requirements():
        print("Some packages failed to install. Please install them manually:")
        print("pip install tensorflow opencv-python numpy scikit-learn matplotlib pillow")
        return
    
    # Setup directories
    setup_directories()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("Next steps:")
    print("1. Download dataset from: https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
    print("2. Save as 'data/raw/garbage-classification.zip'")
    print("3. Run: python extract_and_process.py")
    print("=" * 50)

if __name__ == "__main__":
    main()