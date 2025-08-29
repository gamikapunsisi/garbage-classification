import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

class DataLoader:
    def __init__(self, data_path, img_size=(128, 128)):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = ['plastic', 'organic', 'metal']
        
    def load_data(self):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def create_generators(self, batch_size=32, validation_split=0.2):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        train_generator = datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator