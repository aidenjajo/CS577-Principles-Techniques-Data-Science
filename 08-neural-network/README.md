# Customer Churn Prediction using Artificial Neural Network

Deep learning implementation using TensorFlow/Keras to predict customer churn for a bank based on demographic and account information.

## Overview

This project demonstrates building, training, and evaluating an Artificial Neural Network (ANN) for binary classification. The model predicts whether a bank customer will leave (churn) based on features like credit score, geography, age, balance, and account activity.

## Dataset

**Bank Customer Churn Dataset**
- Samples: 10,000 customers
- Features: 10 attributes (after preprocessing)
- Target: Exited (0 = Stayed, 1 = Left)

**Features:**
- CreditScore: Customer's credit score
- Geography: France, Spain, or Germany
- Gender: Male or Female
- Age: Customer's age
- Tenure: Years with the bank
- Balance: Account balance
- NumOfProducts: Number of bank products
- HasCrCard: Has credit card (1/0)
- IsActiveMember: Active member status (1/0)
- EstimatedSalary: Estimated annual salary

## Technologies

- Python 3.x
- TensorFlow/Keras 2.x (Deep Learning)
- pandas, NumPy
- scikit-learn (preprocessing, metrics)
- matplotlib (visualizations)

## Model Architecture
```
Input Layer (11 features)
    ↓
Hidden Layer 1 (6 neurons, ReLU)
    ↓
Hidden Layer 2 (6 neurons, ReLU)
    ↓
Hidden Layer 3 (6 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Total Parameters:** 163
- Layer 1: 72 parameters (11 inputs × 6 neurons + 6 biases)
- Layer 2: 42 parameters (6 × 6 + 6)
- Layer 3: 42 parameters (6 × 6 + 6)
- Layer 4: 7 parameters (6 × 1 + 1)

## Data Preprocessing

### 1. Feature Extraction
- Used columns [3:-1] from dataset (excludes row number, customer ID, surname)

### 2. Categorical Encoding
- **Gender**: Label Encoding (Male/Female → 0/1)
- **Geography**: One-Hot Encoding with drop='first' (avoids dummy variable trap)

### 3. Train-Test Split
- 80% training, 20% testing
- Random state: 42 for reproducibility

### 4. Feature Scaling
- StandardScaler applied to normalize all features
- Essential for neural network convergence

## Training Configuration

- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Early Stopping**: Patience=10, monitors validation loss
- **Validation Split**: 20% of training data

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.1% |
| Precision (Churn) | 0.7246 |
| Recall (Churn) | 0.5062 |
| F1-Score (Churn) | 0.5957 |

**Confusion Matrix:**
- True Negatives: 1,517 (correctly predicted stayed)
- False Positives: 78 (predicted churn but stayed)
- False Negatives: 200 (predicted stayed but churned)
- True Positives: 205 (correctly predicted churn)

## Usage
```bash
python customer_churn_ann.py
```

## Output

The script generates:
1. Console output with detailed training progress
2. Model architecture summary
3. Training history plot (`training_history.png`)
4. Confusion matrix and performance metrics
5. Sample customer prediction

## Key Concepts

- **Artificial Neural Network**: Multi-layer perceptron for classification
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Backpropagation**: Automatic gradient computation via TensorFlow
- **Early Stopping**: Prevents overfitting by stopping when validation loss plateaus
- **Binary Classification**: Predicting one of two possible outcomes
- **Feature Scaling**: Critical for neural network training convergence

## Implementation Highlights

- Sequential API for straightforward layer stacking
- Early stopping callback to prevent overfitting
- Proper preprocessing pipeline (encoding → scaling)
- Training history visualization for performance monitoring
- Single customer prediction demonstration
- Comprehensive evaluation metrics

## Observations

1. **Model Convergence**: Early stopping triggered around epoch 48, indicating good generalization
2. **Accuracy**: 86.1% test accuracy demonstrates solid performance
3. **Churn Detection**: Model better at identifying customers who stay (95% precision) vs. those who leave (72% precision)
4. **Trade-off**: Lower recall for churn (51%) suggests model is conservative in predicting customer departure

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025
