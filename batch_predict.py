import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_train import GarbageCNN
from torchvision import transforms
from PIL import Image
import os

def batch_predict(image_folder, model_path='models/pytorch_garbage_model.pth', max_images=12):
    """Predict on all images in a folder"""
    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GarbageCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    class_names = ['metal', 'organic', 'plastic']
    
    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(Path(image_folder).rglob(ext))
    
    image_paths = image_paths[:max_images]
    
    if not image_paths:
        print("No images found in the folder!")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process images
    all_images = []
    all_probs = []
    all_preds = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                pred_idx = torch.argmax(probs, dim=0)
            
            # Store results
            all_images.append((img_path, image))
            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred_idx.item())
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Create grid visualization
    n_cols = 4
    n_rows = (len(all_images) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flat
    
    for i, (img_path, image) in enumerate(all_images):
        pred_class = class_names[all_preds[i]]
        confidence = all_probs[i][all_preds[i]]
        
        axes[i].imshow(image)
        axes[i].set_title(f'{pred_class}\n({confidence:.2%})', 
                         color='black', fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(all_images), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Batch Predictions ({len(all_images)} images)', fontsize=16)
    plt.tight_layout()
    plt.savefig('batch_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"Processed {len(all_images)} images:")
    for i, (img_path, _) in enumerate(all_images):
        pred_class = class_names[all_preds[i]]
        confidence = all_probs[i][all_preds[i]]
        print(f"{img_path.name}: {pred_class} ({confidence:.2%})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch prediction on folder')
    parser.add_argument('--folder', type=str, required=True, help='Folder with images')
    parser.add_argument('--model', type=str, default='models/pytorch_garbage_model.pth', 
                       help='Model path')
    parser.add_argument('--max', type=int, default=12, help='Max images to process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return
    
    batch_predict(args.folder, args.model, args.max)

if __name__ == "__main__":
    main()