# CIFAR-10 Image Classification using CNN

Advanced Convolutional Neural Network implementation for multi-class image classification on the CIFAR-10 dataset, achieving 81% test accuracy.

## Overview

This final exam project demonstrates a deep CNN architecture for classifying color images into 10 categories. The model uses batch normalization, dropout regularization, and multiple convolutional blocks to achieve strong performance on the challenging CIFAR-10 benchmark dataset.

## Dataset

**CIFAR-10**
- Training samples: 50,000
- Test samples: 10,000
- Image dimensions: 32×32 pixels (RGB color)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

CIFAR-10 is a widely-used benchmark dataset for evaluating image classification algorithms.

## Technologies

- Python 3.x
- TensorFlow/Keras 2.x
- NumPy
- matplotlib (visualizations)
- scikit-learn (metrics)

## Model Architecture
```
Input (32×32×3)
    ↓
Conv2D Block 1 (32 filters)
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    MaxPool2D (2×2)
    ↓
Conv2D Block 2 (64 filters)
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    MaxPool2D (2×2)
    ↓
Conv2D Block 3 (128 filters)
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    Conv2D (3×3, ReLU, same padding)
    BatchNormalization
    MaxPool2D (2×2)
    ↓
Flatten
    ↓
Dropout (0.2)
    ↓
Dense (128 neurons, ReLU)
    ↓
Dense (10 neurons, Softmax)
```

## Data Preprocessing

1. **Normalization**: Scaled pixel values from [0, 255] to [0, 1]
2. **Label Flattening**: Converted 2D label arrays to 1D format
3. **Train-Validation Split**: 80/20 split for training

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 25
- **Validation Split**: 20%

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 81.38% |
| Training Accuracy | 97.37% |
| Validation Accuracy | 81.48% |
| Test Loss | 0.5421 |
| Total Parameters | ~1.2M |

## Key Architecture Features

1. **Batch Normalization**: Stabilizes learning and speeds up convergence
2. **Multiple Conv Blocks**: Hierarchical feature extraction (32 → 64 → 128 filters)
3. **Same Padding**: Preserves spatial dimensions through convolutions
4. **Dropout (0.2)**: Reduces overfitting in fully connected layers
5. **Progressive Pooling**: Gradual spatial dimension reduction

## Findings and Insights

1. The CNN model consists of stacked Conv2D layers with Batch Normalization and MaxPooling
2. A Dropout layer was added before the Dense layers to reduce overfitting
3. Training accuracy reached 97.37%, while validation accuracy stabilized at ~81.48%
4. This indicates a slightly overfit model, but it still performs well on unseen data
5. Test accuracy of ~81.38% confirms strong generalization capability

## Practical Applications

This type of CNN-based image classification is useful for:
- **Autonomous Vehicles**: Identifying road objects and obstacles
- **Manufacturing**: Detecting defects in production pipelines
- **Agriculture**: Diagnosing plant diseases in fields
- **Mobile Apps**: Enhancing photo sorting and tagging
- **Security**: Object detection in surveillance systems
- **Medical Imaging**: Disease diagnosis from X-rays, MRIs, CT scans

### Example Application: Medical Imaging

This CNN architecture can be adapted for medical imaging systems to analyze X-rays, MRIs, and CT scans for disease detection. The model can identify patterns in medical images to accurately classify patient conditions, supporting healthcare professionals in diagnosis.

## Usage
```bash
python cifar10_classification_cnn.py
```

## Output

The script generates:
1. Console output with training progress
2. Model architecture summary
3. Training history plots (`training_history.png`)
4. Confusion matrix and classification report
5. Saved model file (`cifar10_cnn_model.h5`)

## Implementation Highlights

- Deep architecture with 3 convolutional blocks
- Batch normalization for training stability
- Dropout regularization to prevent overfitting
- Systematic filter progression (32 → 64 → 128)
- Comprehensive evaluation metrics
- Model persistence for future use

## Observations

1. **Strong Performance**: 81% accuracy on complex multi-class color image classification
2. **Slight Overfitting**: Gap between training (97%) and validation (81%) accuracy
3. **Effective Regularization**: Dropout and batch normalization help generalization
4. **Scalable Architecture**: Can be adapted for other image classification tasks
5. **Production Ready**: Model saved in H5 format for deployment

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025 - Final Exam
