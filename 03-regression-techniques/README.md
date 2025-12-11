# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Regression Techniques

Implementation of three fundamental regression methods: Simple Linear, Multiple Linear, and Polynomial Regression with comprehensive visualizations.

## Overview

This homework explores different regression approaches for predicting continuous variables. Part A analyzes salary based on experience, Part B predicts startup profits from multiple factors, and Part C demonstrates when polynomial features outperform linear models.

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (LinearRegression, PolynomialFeatures, OneHotEncoder)
- matplotlib (for visualizations)

## Part A: Simple Linear Regression

**Objective**: Predict salary based on years of experience

**Implementation:**
- Single feature (YearsExperience) to predict target (Salary)
- Linear relationship assumption
- 80/20 train-test split
- Visualizations for both training and test sets

**Key Insight**: Strong linear relationship between experience and salary

## Part B: Multiple Linear Regression

**Objective**: Predict startup profit based on multiple business factors

**Implementation:**
- Multiple features including R&D Spend, Administration, Marketing, and State
- Categorical encoding for State variable using OneHotEncoder
- Handles multiple independent variables simultaneously
- Predictions compared against actual values

**Key Concept**: Demonstrates handling mixed data types (numerical and categorical)

## Part C: Polynomial Regression

**Objective**: Predict salary based on position level using polynomial features

**Datasets:**
- `data_linear_regression.csv` - Experience vs Salary
- `data_multiple_regression.csv` - Startup business metrics
- `data_polynomial_regression.csv` - Position level vs Salary

**Implementation:**
- Compares Linear vs Polynomial (degree 4) regression
- Three visualizations showing progression:
  1. Linear fit (baseline)
  2. Polynomial fit (basic resolution)
  3. Polynomial fit (smooth curve with high resolution)
- Predictions for intermediate value (Level 6.5)

**Key Finding**: Polynomial regression captures non-linear salary growth patterns significantly better than linear regression, especially for executive positions.

## Visualizations

The assignment produces 5 visualizations:
1. Simple Linear Regression - Training Set
2. Simple Linear Regression - Test Set
3. Polynomial vs Linear - Linear Fit
4. Polynomial Regression - Basic Resolution
5. Polynomial Regression - Smooth Curve

## Usage

```bash
python regression_techniques.py
```

## Key Concepts

- **Simple vs Multiple Regression**: Single feature vs multiple features
- **Categorical Encoding**: OneHotEncoder for handling categorical variables
- **Polynomial Features**: Capturing non-linear relationships
- **Model Comparison**: Visual and numerical comparison of different approaches
- **High-Resolution Plotting**: Creating smooth curves for better visualization

## Results

Polynomial regression (degree 4) provides substantially better predictions for position-level salary data compared to linear regression, demonstrating the importance of choosing appropriate model complexity for non-linear relationships.

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025
