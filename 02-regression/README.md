# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Regression Models Comparison

Implementation and comparison of five regression algorithms for predicting electrical power output from a combined cycle power plant based on environmental conditions.

## Overview

This project evaluates multiple regression techniques to determine which model best predicts power plant energy output. The assignment demonstrates understanding of both linear and non-linear regression methods, ensemble techniques, and the importance of feature scaling.

## Dataset

**Combined Cycle Power Plant Dataset**
- Features: Temperature (AT), Vacuum (V), Ambient Pressure (AP), Relative Humidity (RH)
- Target: Electrical Power Output (PE)

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (RandomForestRegressor, DecisionTreeRegressor, LinearRegression, SVR, PolynomialFeatures, StandardScaler)

## Models Implemented

### Part A: Standard Regression Models

1. **Random Forest Regression** - Ensemble method using multiple decision trees for robust predictions
2. **Decision Tree Regression** - Single tree-based model for comparison with ensemble approach
3. **Multiple Linear Regression** - Baseline linear model assuming linear relationships
4. **Polynomial Regression** - Degree 4 polynomial to capture non-linear patterns

### Part B: Support Vector Regression

5. **SVR with RBF Kernel** - Non-linear kernel method requiring feature scaling for optimal performance

## Results Summary

| Model | R² Score | Performance |
|-------|----------|-------------|
| Random Forest | 0.964 | Best overall |
| SVR (RBF) | 0.947 | Strong non-linear fit |
| Polynomial (deg 4) | 0.943 | Good non-linear capture |
| Decision Tree | 0.933 | Solid baseline |
| Linear Regression | 0.930 | Linear baseline |

Random Forest achieved the highest R² score, demonstrating the effectiveness of ensemble methods for this regression task.

## Usage

```bash
python regression_models.py
```

## Key Concepts

- **Model Comparison**: Systematic evaluation of multiple regression algorithms
- **Feature Scaling**: Applied to SVR for distance-based calculations
- **Polynomial Features**: Transforming features to capture non-linear relationships
- **Ensemble Methods**: Combining multiple models to improve prediction accuracy
- **Performance Metrics**: Using MSE and R² to evaluate model quality

## Implementation Notes

- Train-test split: 80/20 with random_state=42 for reproducibility
- Feature scaling applied to both X and y for SVR using StandardScaler
- Polynomial degree of 4 selected for complexity-performance balance

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025
