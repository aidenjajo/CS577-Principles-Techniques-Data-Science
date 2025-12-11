"""
Titanic Survival Prediction - Classification Analysis
CS577-01: Principles and Techniques of Data Science
San Diego State University

This assignment implements multiple classification algorithms to predict
passenger survival on the Titanic disaster using demographic and ticket data.

Models Implemented:
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes

Author: Aiden
Date: October 2024
"""

# ============================================================================
# Import Required Libraries
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Load and Explore Data
# ============================================================================
print("=" * 70)
print("TITANIC SURVIVAL PREDICTION - CLASSIFICATION ANALYSIS")
print("=" * 70)

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

print(f"\nTraining set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# ============================================================================
# PART 2: Data Preprocessing
# ============================================================================
print("\nPreprocessing data...")

# Drop columns with excessive missing values or low predictive power
# Name, Ticket, and Cabin have many missing values or are too specific
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
# Age: Fill with median (common approach for age data)
train_df = train_df.assign(Age=train_df['Age'].fillna(train_df['Age'].median()))
test_df = test_df.assign(Age=test_df['Age'].fillna(test_df['Age'].median()))

# Fare: Fill with median
test_df = test_df.assign(Fare=test_df['Fare'].fillna(test_df['Fare'].median()))

# Embarked: Fill with most common value ('S' = Southampton)
train_df = train_df.assign(Embarked=train_df['Embarked'].fillna('S'))
test_df = test_df.assign(Embarked=test_df['Embarked'].fillna('S'))

# Encode categorical variables
# Sex: Male/Female → 0/1
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
test_df['Sex'] = label_encoder.transform(test_df['Sex'])

# Embarked: C/Q/S → 0/1/2
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'])

print("Preprocessing completed!")

# ============================================================================
# PART 3: Feature and Target Separation
# ============================================================================

# Separate features and target
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (StandardScaler)
# Important for distance-based algorithms (SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")

# ============================================================================
# PART 4: Model Training and Evaluation
# ============================================================================

def train_and_evaluate(model, model_name, use_grid_search=False, param_grid=None):
    """
    Train a model and evaluate its performance.
    
    Parameters:
    - model: Classifier instance
    - model_name: String name for display
    - use_grid_search: Whether to use GridSearchCV
    - param_grid: Hyperparameter grid for GridSearchCV
    """
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    
    # Use GridSearchCV if specified
    if use_grid_search and param_grid:
        print("Performing Grid Search...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
    
    # Training set evaluation
    y_train_pred = best_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Validation set evaluation
    y_val_pred = best_model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("\n--- Validation Set Results ---")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    return best_model, val_accuracy

# ============================================================================
# Define Models with Hyperparameter Grids
# ============================================================================

models_config = {
    "SVM (RBF Kernel)": {
        "model": SVC(),
        "grid_search": True,
        "params": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "grid_search": True,
        "params": {'C': [0.1, 1, 10]}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "grid_search": True,
        "params": {'n_neighbors': [3, 5, 7, 9]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "grid_search": True,
        "params": {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "grid_search": True,
        "params": {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "grid_search": False,
        "params": None
    }
}

# ============================================================================
# Train and Evaluate All Models
# ============================================================================

results = {}

for name, config in models_config.items():
    model, accuracy = train_and_evaluate(
        model=config['model'],
        model_name=name,
        use_grid_search=config['grid_search'],
        param_grid=config['params']
    )
    results[name] = {'model': model, 'accuracy': accuracy}

# ============================================================================
# PART 5: Model Comparison
# ============================================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

# Sort models by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print(f"\n{'Model':<25} {'Validation Accuracy':<20}")
print("-" * 70)
for name, data in sorted_results:
    print(f"{name:<25} {data['accuracy']:.4f}")

# Identify best model
best_model_name = sorted_results[0][0]
best_accuracy = sorted_results[0][1]['accuracy']

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")
print("=" * 70)

print("\nTitanic survival prediction analysis completed!")
