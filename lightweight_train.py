import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_simple_model(input_shape=(64, 64, 3), num_classes=3):
    """Create a very simple CNN model to avoid memory issues"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def lightweight_train():
    """Lightweight training function with smaller images"""
    print("Starting lightweight training...")
    
    if not os.path.exists('data/processed/train'):
        print("Error: Training data not found.")
        return
    
    # Create data generators with smaller images
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Use smaller image size to reduce memory usage
    img_size = (64, 64)
    batch_size = 16  # Smaller batch size
    
    train_generator = datagen.flow_from_directory(
        'data/processed/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        'data/processed/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    print(f"Classes: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create simple model
    model = create_simple_model(input_shape=(img_size[0], img_size[1], 3))
    print("Model created successfully!")
    
    # Train with fewer epochs first
    print("Starting training (5 epochs)...")
    history = model.fit(
        train_generator,
        epochs=5,  # Start with fewer epochs
        validation_data=val_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/lightweight_model.h5')
    print("Model saved as: models/lightweight_model.h5")
    
    return history

if __name__ == "__main__":
    lightweight_train()