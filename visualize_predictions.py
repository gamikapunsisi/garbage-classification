import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from pytorch_train import GarbageDataset, GarbageCNN

def show_batch_predictions(model, dataloader, class_names, n=9):
    """Show batch predictions with true and predicted labels"""
    # Get a batch of data
    model.eval()
    device = next(model.parameters()).device
    
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images[:n])
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    # Convert images back to numpy for plotting
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # CHW to HWC
    images = np.clip(images * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # Denormalize
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flat
    
    for i in range(min(n, len(images))):
        img = images[i]
        true_lbl = class_names[int(labels[i])]
        pred_lbl = class_names[int(preds[i])]
        confidence = probs[i][preds[i]].item()
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl}\nConf: {confidence:.2%}", 
                         color='green' if true_lbl == pred_lbl else 'red',
                         fontsize=10)
        axes[i].axis("off")
    
    # Hide empty subplots
    for i in range(min(n, len(images)), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return images[:n], labels[:n], preds.cpu().numpy(), probs.cpu().numpy()

def create_test_dataloader(batch_size=9):
    """Create test dataloader for visualization"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = GarbageDataset('data/processed/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return test_loader

def load_trained_model(model_path='models/pytorch_garbage_model.pth'):
    """Load the trained model"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = GarbageCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def show_batch_predictions(model, dataloader, class_names, n=9, output_dir="outputs"):
    """Show batch predictions with true and predicted labels and save results"""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images[:n])
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = np.clip(images * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flat
    
    for i in range(min(n, len(images))):
        img = images[i]
        true_lbl = class_names[int(labels[i])]
        pred_lbl = class_names[int(preds[i])]
        confidence = probs[i][preds[i]].item()
        
        # Grid plot
        axes[i].imshow(img)
        axes[i].set_title(f"T:{true_lbl}\nP:{pred_lbl}\nConf:{confidence:.2%}",
                         color='green' if true_lbl == pred_lbl else 'red',
                         fontsize=10)
        axes[i].axis("off")
        
        # Save individual image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        save_name = f"img{i+1}_true-{true_lbl}_pred-{pred_lbl}_conf-{confidence*100:.0f}.png"
        img_pil.save(os.path.join(output_dir, save_name))
    
    for i in range(min(n, len(images)), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    return images[:n], labels[:n], preds.cpu().numpy(), probs.cpu().numpy()


def main():
    print("Loading model and preparing visualization...")
    
    # Load model
    model, device = load_trained_model()
    print(f"Model loaded on device: {device}")
    
    # Create dataloader
    test_loader = create_test_dataloader()
    
    # Class names
    class_names = ['metal', 'organic', 'plastic']
    
    # Show predictions
    print("Generating sample predictions visualization...")
    images, true_labels, predictions, probabilities = show_batch_predictions(
        model, test_loader, class_names, n=9
    )
    
    # Print detailed results
    print("\nDetailed Predictions:")
    print("=" * 60)
    for i in range(len(true_labels)):
        true_class = class_names[int(true_labels[i])]
        pred_class = class_names[int(predictions[i])]
        confidence = probabilities[i][predictions[i]]
        
        status = "✓ CORRECT" if true_class == pred_class else "✗ WRONG"
        print(f"Image {i+1}: {status}")
        print(f"  True: {true_class}")
        print(f"  Pred: {pred_class} ({confidence:.2%})")
        print(f"  All probs: {[f'{p:.2%}' for p in probabilities[i]]}")
        print("-" * 40)

if __name__ == "__main__":
    main()