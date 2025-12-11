# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Titanic Survival Prediction

Classification analysis using multiple machine learning algorithms to predict passenger survival on the Titanic disaster.

## Overview

This assignment applies six different classification algorithms to the classic Titanic dataset. The goal is to predict whether a passenger survived based on features like age, sex, class, and fare. Models are optimized using GridSearchCV and evaluated using confusion matrices and classification reports.

## Dataset

**Titanic - Machine Learning from Disaster**
- Training set: 891 passengers with survival labels
- Test set: 418 passengers
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Target: Survived (0 = No, 1 = Yes)

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (classifiers, GridSearchCV, StandardScaler, LabelEncoder)

## Data Preprocessing

**Missing Value Handling:**
- Age: Filled with median value
- Fare: Filled with median value
- Embarked: Filled with mode ('S' - Southampton)

**Feature Engineering:**
- Dropped: Name, Ticket, Cabin (high missing values or low predictive power)
- Encoded Sex: Male/Female → 0/1 using LabelEncoder
- Encoded Embarked: C/Q/S → 0/1/2 using LabelEncoder

**Scaling:**
- StandardScaler applied to all numerical features

## Models Implemented

All models use GridSearchCV with 5-fold cross-validation:

1. **SVM (RBF Kernel)** - Grid: C = [0.1, 1, 10], gamma = ['scale', 'auto']
2. **Logistic Regression** - Grid: C = [0.1, 1, 10]
3. **K-Nearest Neighbors** - Grid: n_neighbors = [3, 5, 7, 9]
4. **Decision Tree** - Grid: max_depth = [3, 5, 7, 10], min_samples_split = [2, 5, 10]
5. **Random Forest** - Grid: n_estimators = [50, 100], max_depth = [3, 5, 7]
6. **Naive Bayes** - No hyperparameters

## Usage
```bash
python titanic_classification.py
```

## Evaluation Metrics

For each model, the script provides:
- Training accuracy
- Validation accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Key Concepts

- **Data Preprocessing**: Handling missing values, encoding categorical variables
- **Feature Scaling**: StandardScaler for normalized feature ranges
- **GridSearchCV**: Automated hyperparameter tuning
- **Model Comparison**: Systematic evaluation across multiple algorithms
- **Train-Validation Split**: 80/20 split for model evaluation

## Implementation Highlights

- Comprehensive data preprocessing pipeline
- GridSearchCV for optimal hyperparameter selection
- Both training and validation set evaluation
- Detailed performance metrics for each model
- Model comparison and best model identification

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Fall 2024
