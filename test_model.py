import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test_model():
    """Test the trained model on the test set"""
    model_path = 'models/garbage_classifier.h5'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return
    
    # Load model
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'data/processed/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Calculate class-wise accuracy
    class_names = list(test_generator.class_indices.keys())
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    for i in range(len(true_classes)):
        true_class = class_names[true_classes[i]]
        pred_class = class_names[predicted_classes[i]]
        
        class_total[true_class] += 1
        if true_classes[i] == predicted_classes[i]:
            class_correct[true_class] += 1
    
    print("\nClass-wise Accuracy:")
    for class_name in class_names:
        accuracy = class_correct[class_name] / class_total[class_name] if class_total[class_name] > 0 else 0
        print(f"{class_name}: {accuracy:.2%} ({class_correct[class_name]}/{class_total[class_name]})")
    
    return test_accuracy

if __name__ == "__main__":
    test_model()