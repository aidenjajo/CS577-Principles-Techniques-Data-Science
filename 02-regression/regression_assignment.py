"""
Regression Models Comparison Assignment
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden
Date: February 2025
"""

# ============================================================================
# PART A: Random Forest, Decision Tree, Linear, and Polynomial Regression
# ============================================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# Load and Explore Dataset
# ============================================================================
dataset = pd.read_csv("combined_cycle_power_plant.csv")
print("Dataset Head:\n", dataset.head())

# ============================================================================
# Prepare Features and Target Variable
# ============================================================================
# X: All columns except the last (environmental features)
# y: Last column (power output - target variable)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ============================================================================
# Split Dataset into Training and Testing Sets
# ============================================================================
# 80/20 split with random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# Model 1: Random Forest Regression
# ============================================================================
# Ensemble method using multiple decision trees
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ============================================================================
# Model 2: Decision Tree Regression
# ============================================================================
# Single tree-based regression model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# ============================================================================
# Model 3: Multiple Linear Regression
# ============================================================================
# Linear relationship between features and target
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
multi_predictions = multi_model.predict(X_test)

# ============================================================================
# Model 4: Polynomial Regression
# ============================================================================
# Captures non-linear relationships using polynomial features
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
poly_predictions = poly_model.predict(poly.transform(X_test))

# ============================================================================
# Evaluate Part A Models
# ============================================================================
print("\n" + "=" * 70)
print("PART A: MODEL PERFORMANCE COMPARISON")
print("=" * 70)

models = {
    "Random Forest": rf_predictions,
    "Decision Tree": dt_predictions,
    "Linear Regression": multi_predictions,
    "Polynomial Regression": poly_predictions
}

for name, predictions in models.items():
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\n{name} Model:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")

# ============================================================================
# PART B: Support Vector Regression (SVR)
# ============================================================================

# Feature Scaling (required for SVR)
# SVR is sensitive to feature scales, so normalization is essential
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scale training features and fit scaler
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale target variable (reshape for StandardScaler)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# ============================================================================
# Model 5: Support Vector Regression
# ============================================================================
# Uses RBF (Radial Basis Function) kernel for non-linear relationships
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train_scaled)

# ============================================================================
# Predict and Inverse Transform
# ============================================================================
# Predictions are in scaled space, need to transform back to original scale
svr_predictions_scaled = svr_model.predict(X_test_scaled)
svr_predictions = scaler_y.inverse_transform(
    svr_predictions_scaled.reshape(-1, 1)
).flatten()

# ============================================================================
# Evaluate SVR Model
# ============================================================================
print("\n" + "=" * 70)
print("PART B: SVR MODEL PERFORMANCE")
print("=" * 70)
print(f"MSE: {mean_squared_error(y_test, svr_predictions):.4f}")
print(f"R² Score: {r2_score(y_test, svr_predictions):.4f}")
print("=" * 70)
