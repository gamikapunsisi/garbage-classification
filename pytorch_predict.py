import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the same model architecture
class GarbageCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GarbageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PyTorchPredictor:
    def __init__(self, model_path='models/pytorch_garbage_model.pth'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = GarbageCNN(num_classes=3)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        else:
            print("Model not found. Please train first.")
        
        self.class_names = ['metal', 'organic', 'plastic']
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence = confidence.item()
                
                # Get all probabilities
                all_probs = {self.class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
                
            return predicted_class, confidence, all_probs
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None, None
    
    def predict_and_show(self, image_path):
        predicted_class, confidence, probabilities = self.predict(image_path)
        
        if predicted_class is None:
            return
        
        # Display results
        image = Image.open(image_path)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        bars = plt.bar(probabilities.keys(), probabilities.values(), color=colors)
        plt.title('Class Probabilities')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch Garbage Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='models/pytorch_garbage_model.pth', help='Model path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    predictor = PyTorchPredictor(args.model)
    predictor.predict_and_show(args.image)

if __name__ == "__main__":
    main()