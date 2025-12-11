"""
Homework 2: Regression Techniques
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden Jajo
Spring 2025
"""

# ============================================================================
# Import Libraries
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# ============================================================================
# PART A: Simple Linear Regression
# ============================================================================
print("=" * 70)
print("PART A: SIMPLE LINEAR REGRESSION")
print("=" * 70)

# Load Dataset
simple_linear_regression_data = pd.read_csv('data_linear_regression.csv')
X_simple = simple_linear_regression_data[['YearsExperience']].values
y_simple = simple_linear_regression_data['Salary'].values

# Split Dataset into Training and Testing Sets (80/20)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Train Simple Linear Regression Model
simple_linear_regressor = LinearRegression()
simple_linear_regressor.fit(X_train_simple, y_train_simple)

# Visualization 1: Training Set Results
plt.scatter(X_train_simple, y_train_simple, color='red', label='Training Data')
plt.plot(X_train_simple, simple_linear_regressor.predict(X_train_simple), 
         color='blue', label='Regression Line')
plt.title('Simple Linear Regression (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualization 2: Test Set Results
plt.scatter(X_test_simple, y_test_simple, color='green', label='Test Data')
plt.plot(X_train_simple, simple_linear_regressor.predict(X_train_simple), 
         color='blue', label='Regression Line')
plt.title('Simple Linear Regression (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

print("Simple Linear Regression Completed!\n")

# ============================================================================
# PART B: Multiple Linear Regression
# ============================================================================
print("=" * 70)
print("PART B: MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# Load Dataset
multiple_regression_data = pd.read_csv('data_multiple_regression.csv')
X_multiple = multiple_regression_data.iloc[:, :-1].values
y_multiple = multiple_regression_data.iloc[:, -1].values

# Encode Categorical Data (State column at index 3)
# OneHotEncoder transforms categorical variables into binary columns
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], 
    remainder='passthrough'
)
X_multiple = ct.fit_transform(X_multiple)

# Split Dataset into Training and Testing Sets
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple, y_multiple, test_size=0.2, random_state=42
)

# Train Multiple Linear Regression Model
multiple_regressor = LinearRegression()
multiple_regressor.fit(X_train_multiple, y_train_multiple)

# Predict Test Set Results
y_pred_multiple = multiple_regressor.predict(X_test_multiple)

# Display Predictions vs Actual Values
print("\nMultiple Regression Results (Actual vs Predicted):")
print("-" * 70)
for i, (actual, predicted) in enumerate(zip(y_test_multiple, y_pred_multiple), 1):
    print(f"Sample {i}: Actual = ${actual:,.2f}, Predicted = ${predicted:,.2f}")

print("\nMultiple Regression Completed!\n")

# ============================================================================
# PART C: Polynomial Regression
# ============================================================================
print("=" * 70)
print("PART C: POLYNOMIAL REGRESSION")
print("=" * 70)

# Load Dataset
polynomial_regression_data = pd.read_csv('data_polynomial_regression.csv')
X_poly_data = polynomial_regression_data[['Level']].values
y_poly_data = polynomial_regression_data['Salary'].values

# Train Linear Regression Model on Whole Dataset (for comparison)
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_poly_data, y_poly_data)

# Train Polynomial Regression Model (Degree 4)
# Creates polynomial features: x, x^2, x^3, x^4
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X_poly_data)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_poly_data)

# Visualization 3: Linear Regression Fit
plt.scatter(X_poly_data, y_poly_data, color='red', label='Actual Data')
plt.plot(X_poly_data, linear_regressor_poly.predict(X_poly_data), 
         color='blue', label='Linear Fit')
plt.title('Polynomial Regression (Linear Fit)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualization 4: Polynomial Regression (Basic Resolution)
plt.scatter(X_poly_data, y_poly_data, color='red', label='Actual Data')
plt.plot(X_poly_data, poly_regressor.predict(X_poly), 
         color='blue', label='Polynomial Fit')
plt.title('Polynomial Regression (Basic Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualization 5: Polynomial Regression (Smooth Curve - High Resolution)
# Create finer grid for smoother curve visualization
X_grid = np.arange(min(X_poly_data), max(X_poly_data), 0.1).reshape(-1, 1)
plt.scatter(X_poly_data, y_poly_data, color='red', label='Actual Data')
plt.plot(X_grid, poly_regressor.predict(poly_features.transform(X_grid)), 
         color='blue', label='Polynomial Fit (Smooth)')
plt.title('Polynomial Regression (Smooth Fit)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Make Predictions for Position Level 6.5
linear_prediction = linear_regressor_poly.predict([[6.5]])
polynomial_prediction = poly_regressor.predict(poly_features.transform([[6.5]]))

print("\nPredictions for Position Level 6.5:")
print("-" * 70)
print(f"Linear Regression Prediction: ${linear_prediction[0]:,.2f}")
print(f"Polynomial Regression Prediction: ${polynomial_prediction[0]:,.2f}")
print(f"Difference: ${abs(polynomial_prediction[0] - linear_prediction[0]):,.2f}")

print("\nPolynomial Regression Completed!")
print("=" * 70)
