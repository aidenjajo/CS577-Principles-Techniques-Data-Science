"""
Airline Passenger Satisfaction Prediction
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden
Date: October 2024
"""

# ============================================================================
# Import Required Libraries
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Load Dataset
# ============================================================================
print("=" * 70)
print("AIRLINE PASSENGER SATISFACTION PREDICTION")
print("=" * 70)

# Load dataset
df = pd.read_csv("dataset-1.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Target variable: satisfaction")

# ============================================================================
# PART 2: Data Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Remove unnecessary columns
# 'Unnamed: 0' and 'id' are not predictive features
print("\nRemoving unrelated columns...")
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True, errors='ignore')

# Handle outliers in delay columns
# Using IQR method to detect and remove outliers
print("\nHandling outliers in delay columns...")

def remove_outliers_iqr(df, column):
    """Remove outliers using Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    before_count = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after_count = len(df_filtered)
    
    print(f"  {column}: Removed {before_count - after_count} outliers")
    return df_filtered

# Remove outliers from delay columns
df = remove_outliers_iqr(df, 'Departure Delay in Minutes')
df = remove_outliers_iqr(df, 'Arrival Delay in Minutes')

# Handle missing values
print("\nHandling missing values...")
missing_before = df.isnull().sum().sum()

# For numerical columns: fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  {col}: Filled with median")

# For categorical columns: fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  {col}: Filled with mode")

missing_after = df.isnull().sum().sum()
print(f"\nMissing values: {missing_before} → {missing_after}")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()

categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
        print(f"  Encoded: {col}")

print(f"\nFinal dataset shape: {df.shape}")

# ============================================================================
# PART 3: Train-Test Split
# ============================================================================
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

# Separate features and target
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Split dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution in training set:")
print(y_train.value_counts())

# ============================================================================
# PART 4: Model Training with GridSearchCV
# ============================================================================
print("\n" + "=" * 70)
print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("=" * 70)

# Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# XGBoost hyperparameter grid
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# GridSearchCV for Random Forest
print("\nTraining Random Forest with GridSearchCV...")
rf_grid = GridSearchCV(
    rf_model, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)

print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")

# GridSearchCV for XGBoost
print("\nTraining XGBoost with GridSearchCV...")
xgb_grid = GridSearchCV(
    xgb_model, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)

print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best XGBoost CV score: {xgb_grid.best_score_:.4f}")

# Get best models
best_rf = rf_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

# ============================================================================
# PART 5: Model Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION ON TEST SET")
print("=" * 70)

# Random Forest predictions
rf_predictions = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# XGBoost predictions
xgb_predictions = best_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

# Display results
print("\n" + "-" * 70)
print("RANDOM FOREST RESULTS")
print("-" * 70)
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\n" + "-" * 70)
print("XGBOOST RESULTS")
print("-" * 70)
print(f"Accuracy: {xgb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))

# ============================================================================
# PART 6: Visualization
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d',
            cmap='Blues', ax=axes[0])
axes[0].set_title(f'Random Forest Confusion Matrix\nAccuracy: {rf_accuracy:.4f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# XGBoost Confusion Matrix
sns.heatmap(confusion_matrix(y_test, xgb_predictions), annot=True, fmt='d',
            cmap='Greens', ax=axes[1])
axes[1].set_title(f'XGBoost Confusion Matrix\nAccuracy: {xgb_accuracy:.4f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Model Comparison
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(f"\n{'Model':<20} {'Accuracy':<10}")
print("-" * 70)
print(f"{'Random Forest':<20} {rf_accuracy:.4f}")
print(f"{'XGBoost':<20} {xgb_accuracy:.4f}")
print("=" * 70)

print("\nAirline satisfaction prediction analysis completed!")
