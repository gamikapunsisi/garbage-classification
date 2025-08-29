import os
import zipfile
import shutil
import random

def extract_zip():
    """Extract the downloaded zip file"""
    zip_path = 'data/raw/garbage-classification.zip'
    
    if not os.path.exists(zip_path):
        print("Error: Zip file not found at 'data/raw/garbage-classification.zip'")
        print("Please download from: https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
        return False
    
    print("Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/raw/')
        print("Extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False

def organize_data():
    """Organize the extracted data into our structure"""
    print("Organizing data...")
    
    classes = ['plastic', 'organic', 'metal']
    
    # Create target directories
    for class_name in classes:
        os.makedirs(os.path.join('data/processed', class_name), exist_ok=True)
    
    # Map possible folder names to our classes
    class_mapping = {
        'plastic': ['plastic', 'PET', 'HDPE', 'LDPE', 'PP', 'PS'],
        'organic': ['organic', 'food', 'biodegradable', 'compost', 'biological'],
        'metal': ['metal', 'aluminum', 'steel', 'can', 'tin']
    }
    
    processed_count = 0
    source_dir = 'data/raw'
    
    # Walk through extracted files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                
                # Find which class this belongs to
                target_class = None
                for our_class, keywords in class_mapping.items():
                    if any(keyword.lower() in folder_name for keyword in keywords):
                        target_class = our_class
                        break
                
                if target_class:
                    target_dir = os.path.join('data/processed', target_class)
                    target_path = os.path.join(target_dir, file)
                    
                    # Handle duplicates
                    counter = 1
                    while os.path.exists(target_path):
                        name, ext = os.path.splitext(file)
                        target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(file_path, target_path)
                    processed_count += 1
    
    print(f"Organized {processed_count} images into classes")
    return processed_count > 0

def split_data():
    """Split data into train, validation, and test sets without sklearn"""
    print("Splitting data...")
    
    classes = ['plastic', 'organic', 'metal']
    splits = ['train', 'val', 'test']
    
    # Create split directories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join('data/processed', split, class_name), exist_ok=True)
    
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join('data/processed', class_name)
        if not os.path.exists(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(images) == 0:
            continue
            
        total_images += len(images)
        
        # Shuffle images
        random.shuffle(images)
        
        # Split manually: 60% train, 20% validation, 20% test
        train_size = int(0.6 * len(images))
        val_size = int(0.2 * len(images))
        
        train = images[:train_size]
        val = images[train_size:train_size + val_size]
        test = images[train_size + val_size:]
        
        # Move files to appropriate directories
        for split_name, split_files in [('train', train), ('val', val), ('test', test)]:
            for file in split_files:
                src = os.path.join(class_path, file)
                dst = os.path.join('data/processed', split_name, class_name, file)
                shutil.move(src, dst)
        
        # Remove empty class directory
        if os.path.exists(class_path) and not os.listdir(class_path):
            os.rmdir(class_path)
    
    print(f"Split {total_images} images into train/val/test sets")
    return total_images > 0

def main():
    """Main function to process the dataset"""
    print("Starting dataset processing...")
    
    # Step 1: Extract zip
    if not extract_zip():
        return
    
    # Step 2: Organize data
    if not organize_data():
        print("No images found to organize")
        return
    
    # Step 3: Split data
    if not split_data():
        print("No images found to split")
        return
    
    print("Dataset processing completed successfully!")
    print("Data is ready in: data/processed/")

if __name__ == "__main__":
    main()