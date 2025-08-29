import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class GarbagePredictor:
    def __init__(self, model_path='models/garbage_classifier.h5'):
        """Initialize the garbage classifier"""
        self.model = load_model(model_path)
        self.class_names = ['metal', 'organic', 'plastic']
        self.img_size = (128, 128)
        
        print("Model loaded successfully!")
        print(f"Classes: {self.class_names}")
    
    def preprocess_image(self, image_path):
        """Preprocess the input image for prediction"""
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        # Normalize and add batch dimension
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path):
        """Predict the class of an image"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_img)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get all class probabilities
            class_probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            return predicted_class, confidence, class_probabilities
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None, None
    
    def predict_and_visualize(self, image_path):
        """Predict and visualize the results"""
        predicted_class, confidence, probabilities = self.predict(image_path)
        
        if predicted_class is None:
            return
        
        # Read image for visualization
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        
        # Show probabilities
        plt.subplot(1, 2, 2)
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        bars = plt.bar(probabilities.keys(), probabilities.values(), color=colors)
        plt.title('Class Probabilities')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, probabilities.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("All probabilities:")
        for class_name, prob in probabilities.items():
            print(f"  {class_name}: {prob:.2%}")
        
        return predicted_class, confidence

def main():
    """Main function for command line prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict garbage type from image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/garbage_classifier.h5', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        print("Please train the model first with: python simple_train.py")
        return
    
    # Initialize predictor and make prediction
    try:
        predictor = GarbagePredictor(args.model)
        predictor.predict_and_visualize(args.image)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()