import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pytorch_train import GarbageCNN, GarbageDataset

def test_model():
    print("=" * 50)
    print("TESTING PYTORCH MODEL ON TEST SET")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'models/pytorch_garbage_model.pth'
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = GarbageCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = GarbageDataset('data/processed/test', transform=transform)
    
    if len(test_dataset) == 0:
        print("No test data found!")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test the model
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    class_names = ['metal', 'organic', 'plastic']
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-wise accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Calculate accuracies
    overall_accuracy = 100 * correct / total
    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(3)]
    
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print("\nClass-wise Test Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    print(f"\nTotal test samples: {total}")
    
    return overall_accuracy, class_accuracies

if __name__ == "__main__":
    test_model()