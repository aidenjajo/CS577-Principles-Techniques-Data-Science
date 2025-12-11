"""
Data Preprocessing Assignment
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden
Date: February 2025
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================================================
# STEP 2: Load Dataset
# ============================================================================
df = pd.read_csv("Data.csv")

# ============================================================================
# STEP 3: Handle Missing Values
# ============================================================================
# Replace missing values with the mean of each numeric column
df.fillna(df.mean(numeric_only=True), inplace=True)

# ============================================================================
# STEP 4: Encode Categorical Variables
# ============================================================================
# Create a dictionary to store label encoders for each categorical column
label_encoders = {}

# Iterate through all columns and encode categorical (object) types
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# ============================================================================
# STEP 5: Separate Features and Target Variable
# ============================================================================
# X contains all columns except the last one (features/independent variables)
X = df.iloc[:, :-1]
# y contains only the last column (target/dependent variable)
y = df.iloc[:, -1]

# ============================================================================
# STEP 6: Split Dataset into Training and Testing Sets
# ============================================================================
# 80/20 split with random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# STEP 7: Apply Feature Scaling
# ============================================================================
# Initialize StandardScaler for normalization
scaler = StandardScaler()

# Fit the scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler (no fitting on test data)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 8: Display Results
# ============================================================================
print("=" * 70)
print("DATA PREPROCESSING RESULTS")
print("=" * 70)

print("\nFirst 5 rows of scaled training data:")
print("-" * 70)
for i, row in enumerate(X_train_scaled[:5], 1):
    print(f"Row {i}: {row}")

print("\n" + "=" * 70)
print("DATASET DIMENSIONS")
print("=" * 70)
print(f"Training data shape: {X_train_scaled.shape}")
print(f"  - Samples: {X_train_scaled.shape[0]}")
print(f"  - Features: {X_train_scaled.shape[1]}")

print(f"\nTesting data shape: {X_test_scaled.shape}")
print(f"  - Samples: {X_test_scaled.shape[0]}")
print(f"  - Features: {X_test_scaled.shape[1]}")

print("\n" + "=" * 70)
print("ENCODED CATEGORICAL COLUMNS")
print("=" * 70)
if label_encoders:
    for column in label_encoders.keys():
        print(f"  - {column}")
else:
    print("  No categorical columns found in dataset")

print("\n" + "=" * 70)
