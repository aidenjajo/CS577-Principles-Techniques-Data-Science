"""
Fraud Detection with Imbalanced Data Handling
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Train Baseline Model
# ============================================================================
print("=" * 70)
print("FRAUD DETECTION WITH IMBALANCED DATA HANDLING")
print("=" * 70)

# Load Dataset
print("\nLoading the dataset...")
df = pd.read_csv('creditcard.csv')

# Display Dataset Information
print("\n" + "-" * 70)
print("DATASET INFORMATION")
print("-" * 70)
print(f"Dataset Shape: {df.shape}")
print(f"Number of Fraudulent Transactions: {df['Class'].sum()}")
print(f"Number of Legitimate Transactions: {df.shape[0] - df['Class'].sum()}")
print(f"Percentage of Fraudulent Transactions: {df['Class'].mean() * 100:.4f}%")
print("-" * 70)

# Train-Test Split
print("\nSplitting the dataset into training and testing sets...")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Testing set class distribution: {pd.Series(y_test).value_counts().to_dict()}")

# Train Baseline Random Forest Model
print("\n" + "=" * 70)
print("TRAINING BASELINE MODEL")
print("=" * 70)
print("Training Random Forest on imbalanced data...")
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_model.fit(X_train, y_train)

# Evaluate Baseline Model
print("Evaluating the baseline model...")
y_pred_baseline = baseline_model.predict(X_test)
baseline_cm = confusion_matrix(y_test, y_pred_baseline)
baseline_report = classification_report(y_test, y_pred_baseline, output_dict=True)

print("\n" + "-" * 70)
print("BASELINE MODEL RESULTS")
print("-" * 70)
print("\nConfusion Matrix:")
print(baseline_cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))
print("-" * 70)

# ============================================================================
# PART 2: Train SMOTE-Enhanced Model
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SMOTE-ENHANCED MODEL")
print("=" * 70)

# Apply SMOTE to Balance Training Data
print("Applying SMOTE to balance the training dataset...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nOriginal training set shape: {X_train.shape}")
print(f"Resampled training set shape: {X_train_smote.shape}")
print(f"Original training set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Resampled training set class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train SMOTE-Enhanced Random Forest Model
print("\nTraining Random Forest on balanced data...")
smote_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
smote_model.fit(X_train_smote, y_train_smote)

# Evaluate SMOTE-Enhanced Model
print("Evaluating the SMOTE-enhanced model...")
y_pred_smote = smote_model.predict(X_test)
smote_cm = confusion_matrix(y_test, y_pred_smote)
smote_report = classification_report(y_test, y_pred_smote, output_dict=True)

print("\n" + "-" * 70)
print("SMOTE-ENHANCED MODEL RESULTS")
print("-" * 70)
print("\nConfusion Matrix:")
print(smote_cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))
print("-" * 70)

# ============================================================================
# PART 3: Report and Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING COMPARATIVE REPORT")
print("=" * 70)

# Create PDF Report
with PdfPages("model_report.pdf") as pdf:
    # Page 1: Confusion Matrices Comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('Baseline Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(smote_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('SMOTE-Enhanced Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Page 2: Metrics Comparison Table
    metrics = ['precision', 'recall', 'f1-score']
    classes = [0, 1]
    comparison_data = []
    
    for metric in metrics:
        for cls in classes:
            baseline_value = baseline_report[str(cls)][metric]
            smote_value = smote_report[str(cls)][metric]
            comparison_data.append({
                'Metric': f"{metric} (Class {cls})",
                'Baseline': baseline_value,
                'SMOTE-Enhanced': smote_value,
                'Improvement': smote_value - baseline_value
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=comparison_df.values, colLabels=comparison_df.columns, 
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Model Comparison Table", fontweight="bold")
    pdf.savefig()
    plt.close()
    
    # Page 3: Summary Report
    summary = io.StringIO()
    summary.write("=== Summary ===\n")
    summary.write(f"Dataset shape: {df.shape}\n")
    summary.write(f"Fraudulent cases: {df['Class'].sum()} ({df['Class'].mean() * 100:.4f}%)\n\n")
    
    # Calculate improvements
    recall_improvement = (smote_report['1']['recall'] - baseline_report['1']['recall']) * 100
    precision_improvement = (smote_report['1']['precision'] - baseline_report['1']['precision']) * 100
    f1_improvement = (smote_report['1']['f1-score'] - baseline_report['1']['f1-score']) * 100
    
    if recall_improvement > 0:
        summary.write(f"Recall improved by {recall_improvement:.2f}%\n")
    else:
        summary.write("Baseline had better recall\n")
    
    if precision_improvement > 0:
        summary.write(f"Precision improved by {precision_improvement:.2f}%\n")
    else:
        summary.write("Baseline had better precision\n")
    
    if f1_improvement > 0:
        summary.write(f"F1-score improved by {f1_improvement:.2f}%\n")
    else:
        summary.write("Baseline had better F1-score\n")
    
    summary.write("\nObservations:\n")
    summary.write("1. The baseline model has higher precision but lower recall.\n")
    summary.write("2. The SMOTE-enhanced model increases recall and F1-score.\n")
    summary.write("3. Higher recall helps catch more fraud but may increase false positives.\n")
    summary.write("4. Trade-offs depend on business goals.\n")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0, 1, summary.getvalue(), fontsize=10, va="top", family="monospace")
    pdf.savefig()
    plt.close()

print("✅ PDF report saved as 'model_report.pdf'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"\n{'Metric':<25} {'Baseline':<15} {'SMOTE-Enhanced':<15} {'Improvement':<15}")
print("-" * 70)
print(f"{'Fraud Recall':<25} {baseline_report['1']['recall']:.4f}{'':<10} {smote_report['1']['recall']:.4f}{'':<10} +{recall_improvement:.2f}%")
print(f"{'Fraud Precision':<25} {baseline_report['1']['precision']:.4f}{'':<10} {smote_report['1']['precision']:.4f}{'':<10} +{precision_improvement:.2f}%")
print(f"{'Fraud F1-Score':<25} {baseline_report['1']['f1-score']:.4f}{'':<10} {smote_report['1']['f1-score']:.4f}{'':<10} +{f1_improvement:.2f}%")
print("=" * 70)

print("\nFraud detection analysis completed!")
