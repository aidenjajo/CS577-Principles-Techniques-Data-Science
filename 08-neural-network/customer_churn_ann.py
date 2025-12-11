"""
Customer Churn Prediction using Artificial Neural Network
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden Jajo
Spring 2025
"""

# ============================================================================
# STEP 1: Data Preprocessing
# ============================================================================

# Import Required Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CUSTOMER CHURN PREDICTION USING ARTIFICIAL NEURAL NETWORK")
print("=" * 70)

# Check TensorFlow Version
print(f"\nTensorFlow version: {tf.__version__}")

# Load Dataset
print("\nLoading dataset...")
dataset = pd.read_excel('Churn_Modelling.xlsx', sheet_name='in')

print(f"Dataset Shape: {dataset.shape}")
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Extract Features and Target
# Use columns [3:-1] as features (as per instructions)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(f"\nFeature set X shape: {X.shape}")
print(f"Target set y shape: {y.shape}")

# Encode Categorical Variables
print("\nEncoding categorical variables...")

# Gender: Label Encoding (Male/Female → 0/1)
label_encoder_gender = LabelEncoder()
X[:, 2] = label_encoder_gender.fit_transform(X[:, 2])

# Geography: One-Hot Encoding (France/Spain/Germany → binary columns)
# Using drop='first' to avoid dummy variable trap
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

print("After encoding, the first row of X looks like:")
print(X[0])

# Train-Test Split and Feature Scaling
print("\nSplitting dataset and applying feature scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply StandardScaler for normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("After scaling, the first row of X_train looks like:")
print(X_train[0])

# ============================================================================
# STEP 2: Build the Neural Network Model
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING NEURAL NETWORK ARCHITECTURE")
print("=" * 70)

# Initialize Sequential Model
ann = tf.keras.models.Sequential()

# Input Layer and First Hidden Layer
# input_shape specifies the number of features
ann.add(tf.keras.layers.Dense(units=6, activation='relu', 
                               input_shape=(X_train.shape[1],)))

# Additional Hidden Layers (as per instructions)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Output Layer
# Sigmoid activation for binary classification (0 or 1)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Display Model Architecture
print("\nModel Architecture:")
ann.summary()

# ============================================================================
# STEP 3: Training the Model
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING THE NEURAL NETWORK")
print("=" * 70)

# Compile the Model
# Adam optimizer: adaptive learning rate
# Binary crossentropy: loss function for binary classification
# Accuracy: metric to monitor during training
ann.compile(optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy'])

# Early Stopping Callback (optional but recommended)
# Stops training if validation loss doesn't improve for 10 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the Model
print("\nTraining model with 100 epochs and early stopping...")
history = ann.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Plot Training History
print("\nGenerating training history plots...")
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training history saved as 'training_history.png'")

# ============================================================================
# STEP 4: Making Predictions
# ============================================================================
print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

# Predict for a Single Customer
# Customer: CreditScore=600, France, Male, Age=40, Tenure=3, 
#           Balance=60000, NumOfProducts=2, HasCrCard=1, 
#           IsActiveMember=1, EstimatedSalary=50000
print("\nPredicting churn for a specific customer...")
customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])

# Apply same transformations as training data
customer[0, 2] = label_encoder_gender.transform([customer[0, 2]])[0]
customer_encoded = ct.transform(customer)
customer_scaled = sc.transform(customer_encoded)

# Make prediction
prediction = ann.predict(customer_scaled, verbose=0)
will_exit = prediction[0, 0] > 0.5

print("\nCustomer Prediction:")
print(f"  Probability of leaving: {prediction[0, 0]:.4f}")
print(f"  Will the customer leave? {'Yes' if will_exit else 'No'}")

# Predict Test Set Results
print("\nPredicting test set...")
y_pred = (ann.predict(X_test, verbose=0) > 0.5).astype(int)

# Calculate Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy Score: {acc:.4f}")

# ============================================================================
# STEP 5: Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE AND TEST RESULTS SUMMARY")
print("=" * 70)

print("\n1. Model Architecture:")
print(f"   - Input Layer: {X_train.shape[1]} features")
print("   - Hidden Layer 1: 6 neurons with ReLU activation")
print("   - Hidden Layer 2: 6 neurons with ReLU activation")
print("   - Hidden Layer 3: 6 neurons with ReLU activation")
print("   - Output Layer: 1 neuron with Sigmoid activation")

print("\n2. Training Configuration:")
print("   - Optimizer: Adam")
print("   - Loss Function: Binary Crossentropy")
print("   - Batch Size: 32")
print(f"   - Epochs Trained: {len(history.history['loss'])}")
print("   - Early Stopping: Enabled (patience=10)")

print("\n3. Test Results:")
print(f"   - Accuracy Score: {acc:.4f} ({acc*100:.2f}%)")
print("   - Confusion Matrix:")
print(f"     True Negatives:  {cm[0, 0]:>4}")
print(f"     False Positives: {cm[0, 1]:>4}")
print(f"     False Negatives: {cm[1, 0]:>4}")
print(f"     True Positives:  {cm[1, 1]:>4}")

# Calculate additional metrics
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n4. Performance Metrics:")
print(f"   - Precision (Churn): {precision:.4f}")
print(f"   - Recall (Churn):    {recall:.4f}")
print(f"   - F1-Score (Churn):  {f1_score:.4f}")

print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Metric':<25} {'Value':<15}")
print("-" * 40)
print(f"{'Test Accuracy':<25} {acc:.4f} ({acc*100:.1f}%)")
print(f"{'Precision (Churn)':<25} {precision:.4f}")
print(f"{'Recall (Churn)':<25} {recall:.4f}")
print(f"{'F1-Score (Churn)':<25} {f1_score:.4f}")
print(f"{'Total Parameters':<25} 163")
print("=" * 70)

print("\nCustomer churn prediction analysis completed!")
