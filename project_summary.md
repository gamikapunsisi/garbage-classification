# Garbage Classification Project Summary

## Problem Statement
Develop a computer vision system to classify waste materials into three categories (plastic, organic, metal) to aid in automated recycling systems.

## Approach

### Technology Stack
- **PyTorch**: For building and training the CNN model (chosen for better macOS compatibility)
- **TorchVision**: For image preprocessing and transformations
- **PIL/Pillow**: For image handling
- **Matplotlib**: For visualization

### Model Architecture
- Convolutional Neural Network (CNN) with 3 convolutional layers
- MaxPooling layers for dimensionality reduction
- Fully connected layers for classification
- Dropout for regularization (0.5)

### Data Processing
- Image resizing to 128x128 pixels
- Normalization (ImageNet standards)
- Train/Validation/Test split (60%/20%/20%)

## Results

### Training Performance
- **Final Training Accuracy**: 90.77%
- **Final Validation Accuracy**: 84.32%
- **Best Validation Accuracy**: 86.04% (epoch 9)
- **Training Loss**: 0.2304
- **Validation Loss**: 0.4434

### Test Performance
- **Overall Test Accuracy**: [To be filled after testing]
- **Class-wise Accuracy**:
  - Metal: [To be filled]
  - Organic: [To be filled] 
  - Plastic: [To be filled]

## Model Details
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Cross Entropy Loss
- **Batch Size**: 32
- **Epochs**: 10
- **Device**: MPS (Apple Silicon Metal Performance Shaders)

## Challenges Faced
1. **TensorFlow Compatibility**: Segmentation faults on macOS led to switching to PyTorch
2. **Dataset Organization**: Manual processing of Kaggle dataset structure
3. **Class Imbalance**: Handling different number of images per category
4. **Hardware Limitations**: Training on consumer hardware

## Usage

### Training
```bash
python pytorch_train.py