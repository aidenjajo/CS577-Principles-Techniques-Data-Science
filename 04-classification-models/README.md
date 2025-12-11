# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Classification Models Comparison

Comprehensive comparison of seven classification algorithms for predicting customer purchasing behavior based on demographic features.

## Overview

This assignment evaluates multiple classification techniques using GridSearchCV for hyperparameter tuning. The goal is to predict whether a customer will purchase a product based on their age and estimated salary from social network advertising data.

## Dataset

**Social Network Ads Classification**
- Features: Age, Estimated Salary
- Target: Purchased (0 = No, 1 = Yes)
- Train-Test Split: 80/20 with random_state=0

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (classifiers, GridSearchCV, metrics, StandardScaler)
- matplotlib, seaborn (visualizations)
- mlxtend (decision boundary visualization)

## Models Implemented

All models use GridSearchCV with 5-fold cross-validation for hyperparameter tuning:

1. **Logistic Regression** - Parameter grid: C = [0.1, 1, 10]
2. **K-Nearest Neighbors (KNN)** - Parameter grid: n_neighbors = [3, 5, 7]
3. **SVM Linear** - Parameter grid: C = [0.1, 1, 10]
4. **SVM RBF** - Parameter grid: C = [0.1, 1, 10], gamma = ['scale', 'auto']
5. **Decision Tree** - Parameter grid: max_depth = [3, 5, 10]
6. **Random Forest** - Parameter grid: n_estimators = [50, 100], max_depth = [3, 5, 10]
7. **Naive Bayes** - No hyperparameters

## Results Summary

| Model | Accuracy |
|-------|----------|
| KNN | 95.0% |
| SVM RBF | 95.0% |
| Decision Tree | 95.0% |
| Random Forest | 95.0% |
| Logistic Regression | 92.5% |
| SVM Linear | 91.25% |
| Naive Bayes | 91.25% |

**Best Performers**: KNN, SVM RBF, Decision Tree, and Random Forest all achieved 95% accuracy on the test set.

## Visualizations

The assignment produces comprehensive visualizations:

1. **Confusion Matrices** - Heatmaps for all 7 models showing TP, TN, FP, FN
2. **Decision Boundaries** - Visual representation of classification regions for each model
3. **Scatter Plots** - Training and test results with correct/incorrect predictions color-coded
   - Green: Correct predictions
   - Red: Incorrect predictions

## Usage
```bash
python classification_comparison.py
```

## Key Concepts

- **GridSearchCV**: Automated hyperparameter tuning with cross-validation
- **Feature Scaling**: StandardScaler applied for distance-based algorithms
- **Confusion Matrix**: Detailed breakdown of classification performance
- **Decision Boundaries**: Visual understanding of model classification logic
- **Model Comparison**: Systematic evaluation across multiple algorithms

## Implementation Highlights

- Feature scaling applied to training data, then transformed to test data (prevents data leakage)
- GridSearchCV with 5-fold cross-validation for robust parameter selection
- Comprehensive evaluation using confusion matrices and classification reports
- Multiple visualization techniques for result interpretation

## Key Findings

Multiple models (KNN, SVM RBF, Decision Tree, Random Forest) achieved optimal 95% accuracy, demonstrating that this dataset is well-suited for various classification approaches. The similar performance suggests the decision boundary is relatively clear and can be captured by different algorithmic strategies.

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Fall 2024
