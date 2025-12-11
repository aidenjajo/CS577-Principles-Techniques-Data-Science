"""
Classification Models Comparison
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden Jajo
Spring 2025
"""

# ============================================================================
# Import Required Libraries
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

# ============================================================================
# PART 1: Data Preparation
# ============================================================================

# Load Dataset
data = pd.read_csv("Social_Network_Ads_Classification.csv")
features = data[['Age', 'EstimatedSalary']]
target = data['Purchased']

# Split Dataset into Training and Test Sets
# 80/20 split with random_state=0 as per instructions
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=0
)

# Apply Feature Scaling (StandardScaler)
# Required for distance-based algorithms like KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PART 2: Model Training with GridSearchCV
# ============================================================================

# Define models and hyperparameter grids
models = {
    "LogReg": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "SVM Linear": (SVC(kernel='linear'), {'C': [0.1, 1, 10]}),
    "SVM RBF": (SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [50, 100], 
        'max_depth': [3, 5, 10]
    }),
    "Naive Bayes": (GaussianNB(), {}),
}

# Storage for best models and performance metrics
best_models = {}
performance = {}

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Train models using GridSearchCV and evaluate performance.
    
    Parameters:
    - models: Dictionary of model names, estimators, and parameter grids
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    """
    for name, (model, params) in models.items():
        # Perform grid search with 5-fold cross-validation
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        # Store best model
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        # Make predictions on test set
        y_pred = best_model.predict(X_test)
        
        # Store evaluation metrics
        performance[name] = {
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "accuracy": accuracy_score(y_test, y_pred),
            "y_pred": y_pred
        }

# Execute training
print("Training models with GridSearchCV...")
train_and_evaluate(models, X_train_scaled, y_train, X_test_scaled, y_test)
print("Training completed!\n")

# ============================================================================
# PART 3: Model Evaluation and Visualization
# ============================================================================

def plot_confusion_matrices(results):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(results.items()):
        sns.heatmap(data['conf_matrix'], annot=True, fmt='d', 
                   ax=axes[i], cmap='coolwarm')
        axes[i].set_title(f'{name} - Accuracy: {data["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_decision_boundaries(models, X_train, y_train):
    """Visualize decision boundaries for each classifier."""
    for name, model in models.items():
        if hasattr(model, "predict"):
            plt.figure(figsize=(8, 6))
            plot_decision_regions(X_train, y_train.to_numpy(), clf=model, legend=2)
            plt.title(f"Decision Boundary - {name}")
            plt.xlabel("Age (Scaled)")
            plt.ylabel("Estimated Salary (Scaled)")
            plt.show()

def print_classification_reports(results):
    """Print detailed classification reports for all models."""
    for name, data in results.items():
        print(f"\n{'=' * 70}")
        print(f"{name} Classification Report:")
        print('=' * 70)
        print(pd.DataFrame(data['report']).transpose())

def compare_models(results):
    """Compare model performance and display sorted results."""
    perf_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [data["accuracy"] for data in results.values()]
    })
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(perf_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
    print("=" * 70)

def plot_results_scatter(X, y_true, y_pred, title):
    """
    Scatter plot showing TP, FP, FN, TN in different colors.
    
    Green: Correct predictions
    Red: Incorrect predictions
    """
    plt.figure(figsize=(8, 6))
    colors = ['green' if yt == yp else 'red' for yt, yp in zip(y_true, y_pred)]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolors='k')
    plt.xlabel("Age (Scaled)")
    plt.ylabel("Estimated Salary (Scaled)")
    plt.title(title)
    plt.legend(['Green = Correct', 'Red = Incorrect'])
    plt.show()

# ============================================================================
# Generate All Visualizations and Reports
# ============================================================================

print("\nGenerating visualizations...\n")

# Scatter plots for training and test results
for name, model in best_models.items():
    y_train_pred = model.predict(X_train_scaled)
    plot_results_scatter(X_train_scaled, y_train, y_train_pred, 
                        f"Training Results - {name}")
    plot_results_scatter(X_test_scaled, y_test, performance[name]['y_pred'], 
                        f"Test Results - {name}")

# Confusion matrices
plot_confusion_matrices(performance)

# Decision boundaries
visualize_decision_boundaries(best_models, X_train_scaled, y_train)

# Classification reports
print_classification_reports(performance)

# Model comparison
compare_models(performance)

print("\nClassification analysis completed!")
