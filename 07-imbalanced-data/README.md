# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Fraud Detection with Imbalanced Data Handling

Demonstration of SMOTE (Synthetic Minority Over-sampling Technique) for handling severely imbalanced datasets in credit card fraud detection.

## Overview

This assignment addresses the challenge of class imbalance in fraud detection, where fraudulent transactions represent only 0.26% of the dataset. By comparing a baseline model trained on imbalanced data against a SMOTE-enhanced model trained on balanced data, the project demonstrates the importance of proper handling of imbalanced datasets.

## Dataset

**Credit Card Fraud Detection**
- Total Transactions: 39,999
- Fraudulent: 104 (0.26%)
- Legitimate: 39,895 (99.74%)
- Features: 30 anonymized features (V1-V28, Time, Amount)
- Target: Class (0 = Normal, 1 = Fraud)

**Severe Imbalance**: Fraud cases represent only 1 in 384 transactions

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (RandomForestClassifier, train_test_split, metrics)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn (visualizations)

## Problem: Class Imbalance

Traditional machine learning algorithms struggle with imbalanced datasets because they optimize for overall accuracy, which can be achieved by simply predicting the majority class. In fraud detection, this results in:
- High overall accuracy (99.7%+) but poor fraud detection
- Low recall for fraudulent transactions (missing actual fraud)
- Biased model favoring the majority class

## Solution: SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)**
- Creates synthetic samples of the minority class (fraud)
- Balances the training dataset without losing information
- Improves model's ability to learn fraud patterns

## Models Implemented

### 1. Baseline Model
- Algorithm: Random Forest (100 estimators)
- Training Data: Original imbalanced dataset
- Purpose: Demonstrate poor performance on imbalanced data

### 2. SMOTE-Enhanced Model
- Algorithm: Random Forest (100 estimators)
- Training Data: SMOTE-balanced dataset (50/50 split)
- Purpose: Show improvement with balanced training data

## Results

| Metric (Fraud Class) | Baseline | SMOTE-Enhanced | Improvement |
|----------------------|----------|----------------|-------------|
| Recall | 76.19% | 95.24% | **+19.05%** |
| Precision | 94.12% | 95.24% | +1.12% |
| F1-Score | 84.21% | 95.24% | **+11.03%** |

**Key Finding**: SMOTE improved fraud detection recall by 19%, meaning the model catches 95% of fraud cases compared to only 76% with the baseline.

## Confusion Matrix Comparison

**Baseline Model:**
- True Negatives: 7,978 | False Positives: 1
- False Negatives: 5 (missed fraud) | True Positives: 16

**SMOTE-Enhanced Model:**
- True Negatives: 7,978 | False Positives: 1
- False Negatives: 1 (missed fraud) | True Positives: 20

The SMOTE model catches 4 additional fraud cases while maintaining the same false positive rate.

## Usage
```bash
python fraud_detection_smote.py
```

## Output

The script generates:
1. Console output with detailed metrics
2. PDF report (`model_report.pdf`) containing:
   - Confusion matrix visualizations
   - Metrics comparison table
   - Summary of results and observations

## Key Concepts

- **Class Imbalance**: When one class significantly outnumbers another
- **SMOTE**: Synthetic minority over-sampling technique
- **Recall vs Precision Trade-off**: In fraud detection, recall (catching fraud) is often more important than precision
- **Stratified Sampling**: Maintaining class distribution in train-test split
- **Synthetic Data Generation**: Creating realistic minority class samples

## Implementation Highlights

- Stratified train-test split preserves class imbalance in both sets
- SMOTE applied only to training data (prevents data leakage)
- Balanced dataset: Original 83 fraud cases → 31,916 synthetic fraud cases
- Comprehensive evaluation with confusion matrices and classification reports
- Automated PDF report generation for documentation

## Observations

1. **Baseline Limitations**: High precision but low recall means many fraud cases are missed
2. **SMOTE Benefits**: Dramatically improves recall while maintaining precision
3. **Business Impact**: Catching 95% vs 76% of fraud saves significant financial losses
4. **Trade-offs**: Slight increase in false positives is acceptable for better fraud detection

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Fall 2024
