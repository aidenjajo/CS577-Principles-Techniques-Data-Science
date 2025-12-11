# MNIST Digit Classification using CNN

Implementation of a Convolutional Neural Network for handwritten digit recognition on the MNIST dataset, achieving ~94% test accuracy.

## Overview

This project demonstrates fundamental concepts in deep learning for computer vision using CNNs. The model learns to recognize handwritten digits (0-9) through convolutional feature extraction and classification.

## Dataset

**MNIST Handwritten Digits**
- Training samples: 60,000
- Test samples: 10,000
- Image dimensions: 28×28 pixels (grayscale)
- Classes: 10 (digits 0-9)

The MNIST dataset is a standard benchmark for image classification algorithms.

## Technologies

- Python 3.x
- TensorFlow/Keras 2.x
- NumPy
- matplotlib (visualizations)

## Model Architecture
```
Input (28×28×1)
    ↓
Conv2D (4 filters, 3×3, ReLU)
    ↓
MaxPool2D (2×2)
    ↓
Conv2D (4 filters, 3×3, ReLU)
    ↓
MaxPool2D (2×2)
    ↓
Flatten
    ↓
Dense (64 neurons, ReLU)
    ↓
Dense (10 neurons, Softmax)
```

**Total Parameters:** ~2,500

## Data Preprocessing

1. **Reshape**: Added channel dimension (28, 28, 1)
2. **Normalization**: Scaled pixel values from [0, 255] to [0, 1]
3. **Train-Validation Split**: 80/20 split for training

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 2,048
- **Epochs**: 15 (with early stopping, patience=3)
- **Validation Split**: 20%

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.9% |
| Training Accuracy | 95.2% |
| Validation Accuracy | 94.5% |
| Test Loss | 0.2134 |
| Epochs Trained | 12 |

## Key CNN Concepts Demonstrated

1. **Convolutional Layers**: Extract spatial features through learnable filters
2. **Pooling Layers**: Reduce spatial dimensions while retaining important features
3. **Feature Maps**: Hierarchical feature learning (edges → shapes → digits)
4. **Flattening**: Convert 2D feature maps to 1D for dense layers
5. **Softmax Classification**: Probability distribution over 10 digit classes

## Usage
```bash
python mnist_digit_classification_cnn.py
```

## Output

The script generates:
1. Console output with training progress
2. Model architecture summary
3. Training history plots (`training_history_cnn.png`)
4. Sample predictions visualization (`sample_predictions.png`)
5. Performance metrics and summary

## Implementation Highlights

- Efficient architecture balancing complexity and computational cost
- Early stopping prevents overfitting
- Visualization of training progress (accuracy and loss curves)
- Sample predictions showing model performance on individual images
- Comprehensive evaluation metrics

## Observations

1. **Quick Convergence**: Model achieves >85% accuracy within first few epochs
2. **Minimal Overfitting**: Training and validation curves track closely
3. **Feature Extraction**: Convolutional layers effectively capture digit patterns
4. **Computational Efficiency**: Lightweight architecture trains quickly on CPU/GPU
5. **Strong Generalization**: High test accuracy demonstrates good generalization

## Comparison: CNN vs Traditional Methods

CNNs outperform traditional ML methods (SVM, Random Forest) on image data because:
- Automatic feature learning (no manual feature engineering)
- Spatial hierarchy preservation
- Translation invariance through pooling
- Parameter sharing reduces model complexity

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025
