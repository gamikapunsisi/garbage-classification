import os
import shutil
import random
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        self.classes = ['plastic', 'organic', 'metal']
        
    def organize_kaggle_data(self):
        """
        Reorganize Kaggle Garbage Classification dataset into our required structure
        """
        # Create target directories
        for class_name in self.classes:
            os.makedirs(os.path.join(self.target_path, class_name), exist_ok=True)
        
        # Map Kaggle classes to our classes for the specific dataset
        class_mapping = {
            'plastic': ['plastic', 'PET', 'HDPE', 'LDPE', 'PP', 'PS', 'Plastic'],
            'organic': ['organic', 'food', 'biodegradable', 'compost', 'Organic', 'biological'],
            'metal': ['metal', 'aluminum', 'steel', 'can', 'Metal', 'tin']
        }
        
        # Process each file in the source directory
        processed_count = 0
        skipped_count = 0
        
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Determine class based on folder name
                    folder_name = os.path.basename(root).lower()
                    
                    target_class = None
                    for our_class, kaggle_classes in class_mapping.items():
                        if any(kaggle_class.lower() in folder_name for kaggle_class in kaggle_classes):
                            target_class = our_class
                            break
                    
                    if target_class:
                        # Copy file to appropriate folder
                        target_dir = os.path.join(self.target_path, target_class)
                        target_file_path = os.path.join(target_dir, file)
                        
                        # Handle duplicate filenames
                        counter = 1
                        while os.path.exists(target_file_path):
                            name, ext = os.path.splitext(file)
                            target_file_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        shutil.copy2(file_path, target_file_path)
                        processed_count += 1
                    else:
                        skipped_count += 1
        
        print(f"Processed {processed_count} images, skipped {skipped_count} images")
        
        # Check if we have enough data in each class
        for class_name in self.classes:
            class_path = os.path.join(self.target_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"Class '{class_name}': {count} images")
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets
        """
        # Create split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            split_path = os.path.join(self.target_path, split)
            for class_name in self.classes:
                os.makedirs(os.path.join(split_path, class_name), exist_ok=True)
        
        # Split each class
        total_images = 0
        for class_name in self.classes:
            class_path = os.path.join(self.target_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(images) == 0:
                continue
                
            total_images += len(images)
            
            # First split: train + temp, then split temp into val + test
            train, temp = train_test_split(images, test_size=test_size + val_size, random_state=42)
            val, test = train_test_split(temp, test_size=test_size/(test_size + val_size), random_state=42)
            
            # Move files to appropriate directories
            for split_name, split_files in [('train', train), ('val', val), ('test', test)]:
                for file in split_files:
                    src = os.path.join(class_path, file)
                    dst = os.path.join(self.target_path, split_name, class_name, file)
                    shutil.move(src, dst)
            
            # Remove empty class directory
            if os.path.exists(class_path) and not os.listdir(class_path):
                os.rmdir(class_path)
        
        print(f"Total images processed: {total_images}")
        print("Data split completed successfully!")

# Alternative function for direct usage
def download_and_process_data():
    import kagglehub
    
    print("Downloading dataset...")
    path = kagglehub.dataset_download("mostafaabla/garbage-classification")
    print(f"Dataset downloaded to: {path}")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Preprocess directly from downloaded path
    preprocessor = DataPreprocessor(path, 'data/processed')
    preprocessor.organize_kaggle_data()
    preprocessor.split_data()
    
    return 'data/processed'

if __name__ == "__main__":
    data_path = download_and_process_data()
    print(f"Data is ready at: {data_path}")