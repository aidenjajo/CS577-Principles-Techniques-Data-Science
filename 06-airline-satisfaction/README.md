# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Airline Passenger Satisfaction Prediction

Advanced classification analysis predicting airline passenger satisfaction using Random Forest and XGBoost with comprehensive data preprocessing and hyperparameter optimization.

## Overview

This project analyzes airline passenger survey data to predict satisfaction levels. The assignment demonstrates advanced preprocessing techniques including outlier removal using IQR, strategic missing value imputation, and categorical encoding. Both Random Forest and XGBoost models are optimized using GridSearchCV.

## Dataset

**Airline Passenger Satisfaction Survey**
- Samples: ~100,000+ passengers
- Features: 23 attributes including demographics, flight details, and service ratings
- Target: Satisfaction (satisfied / neutral or dissatisfied)

**Key Features:**
- Demographics: Gender, Age, Customer Type
- Flight Details: Class, Type of Travel, Flight Distance
- Service Ratings: WiFi, Food, Seat Comfort, Entertainment, Cleanliness (1-5 scale)
- Delays: Departure and Arrival Delay in Minutes

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (RandomForestClassifier, GridSearchCV, LabelEncoder)
- XGBoost (XGBClassifier)
- matplotlib, seaborn (visualizations)

## Data Preprocessing

### 1. Column Removal
- Dropped: `Unnamed: 0`, `id` (non-predictive identifiers)

### 2. Outlier Handling
- Method: Interquartile Range (IQR)
- Applied to: Departure Delay and Arrival Delay columns
- Formula: Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] removed

### 3. Missing Value Imputation
- Numerical features: Filled with median
- Categorical features: Filled with mode (most frequent value)

### 4. Categorical Encoding
- LabelEncoder applied to: Gender, Customer Type, Type of Travel, Class, Satisfaction
- Converts categorical variables to numerical representations

## Models Implemented

### Random Forest Classifier
**Hyperparameter Grid:**
- n_estimators: [100, 200]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5]
- min_samples_leaf: [1, 2]

### XGBoost Classifier
**Hyperparameter Grid:**
- n_estimators: [100, 200]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.1]
- subsample: [0.8, 1.0]

Both models use GridSearchCV with 3-fold cross-validation for optimal parameter selection.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 96% | 0.96 | 0.96 | 0.96 |
| XGBoost | 96% | 0.96 | 0.96 | 0.96 |

Both models achieved excellent 96% accuracy, demonstrating robust prediction of passenger satisfaction.

## Usage
```bash
python airline_satisfaction.py
```

## Visualizations

The analysis produces:
- Confusion matrices for both models (side-by-side comparison)
- Classification reports with precision, recall, and F1-scores
- Model performance comparison table

## Key Concepts

- **Outlier Detection**: IQR method for identifying and removing statistical outliers
- **Missing Value Strategies**: Median for numerical, mode for categorical data
- **Hyperparameter Tuning**: GridSearchCV for systematic optimization
- **Ensemble Methods**: Random Forest for robust predictions
- **Gradient Boosting**: XGBoost for high-performance classification
- **Model Comparison**: Systematic evaluation of multiple approaches

## Implementation Highlights

- Comprehensive preprocessing pipeline addressing data quality issues
- Strategic handling of delay outliers to improve model generalization
- GridSearchCV with cross-validation for unbiased hyperparameter selection
- Equal performance from both models validates preprocessing effectiveness
- Production-ready code with error handling and clear documentation

## Key Findings

Both Random Forest and XGBoost achieved identical 96% accuracy, suggesting the preprocessing pipeline effectively prepared the data for multiple algorithmic approaches. The models successfully identify key factors influencing passenger satisfaction, with service quality ratings and customer type being strong predictors.

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Fall 2024
