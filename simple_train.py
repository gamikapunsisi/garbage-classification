import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_model(input_shape=(128, 128, 3), num_classes=3):
    """Create a simple CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the garbage classification model"""
    print("=" * 50)
    print("GARBAGE CLASSIFICATION MODEL TRAINING")
    print("=" * 50)
    
    print("Checking data directory...")
    
    if not os.path.exists('data/processed/train'):
        print("Error: Training data not found. Please run extract_and_process.py first.")
        return
    
    # Create data generators
    print("Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    print(f"Classes: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create and train model
    print("Creating model...")
    model = create_model()
    print("Model summary:")
    model.summary()
    
    print("Starting training...")
    print("This may take a while...")
    
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/garbage_classifier.h5')
    print("Model saved as: models/garbage_classifier.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as: training_history.png")
    
    # Show final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Ask if user wants to show the plot
    show_plot = input("\nShow training history plot? (y/n): ").lower().strip()
    if show_plot == 'y':
        plt.show()
    
    return history

if __name__ == "__main__":
    train_model()