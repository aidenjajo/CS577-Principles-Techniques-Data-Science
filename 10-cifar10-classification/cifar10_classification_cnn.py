"""
CIFAR-10 Image Classification using CNN
CS577-01: Principles and Techniques of Data Science
San Diego State University - Final Exam
Author: Aiden Jajo
Spring 2025
"""

# ============================================================================
# 1) Data Preprocess
# ============================================================================

# a) Import the necessary Python libraries for data processing, model creation, and evaluation.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 70)
print("CIFAR-10 IMAGE CLASSIFICATION USING CNN")
print("=" * 70)

# b) Load the CIFAR-10 dataset using TensorFlow's built-in datasets module.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"\nTraining data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")

# c) Normalize the pixel values to scale them between 0 and 1, which improves training performance.
x_train = x_train / 255.0
x_test = x_test / 255.0

# d) Flatten the label arrays so they are in a 1D format suitable for training.
y_train = y_train.flatten()
y_test = y_test.flatten()

# ============================================================================
# 2) Sequential Model
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

# a) Initialize a Sequential model using Keras.
model = keras.Sequential()

# b) Add an input layer that matches the shape of the CIFAR-10 images (32x32x3).
model.add(layers.InputLayer(input_shape=(32, 32, 3)))

# c) Add the first convolutional block with Conv2D, Batch Normalization, and MaxPooling.
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# d) Add the second convolutional block with increased filters and the same structure.
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# e) Add the third convolutional block with further increased filters for deeper feature extraction.
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# f) Flatten the feature maps, apply dropout to reduce overfitting, and add fully connected layers for classification.
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print("\nModel Summary:")
model.summary()

# ============================================================================
# 3) Training
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

# Compile the model using the Adam optimizer and sparse categorical crossentropy loss.
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model on the training set for 25 epochs using a validation split of 20%.
train_history = model.fit(x_train, y_train, batch_size=32, epochs=25, validation_split=0.2)

# ============================================================================
# 4) Evaluating
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

# a) Plot the training and validation accuracy over each epoch to visualize learning progress.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_history.history['sparse_categorical_accuracy'], label='Train Acc')
plt.plot(train_history.history['val_sparse_categorical_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# b) Plot the training and validation loss over each epoch to monitor model convergence.
plt.subplot(1, 2, 2)
plt.plot(train_history.history['loss'], label='Train Loss')
plt.plot(train_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# c) Evaluate the model's performance on the test dataset and obtain loss and accuracy.
evaluate_history = model.evaluate(x_test, y_test)

# d) Generate class predictions for the test dataset using the trained model.
y_pred = np.argmax(model.predict(x_test), axis=1)

# e) Print the confusion matrix and classification report to analyze classification performance.
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================================
# 5) Summary
# ============================================================================
print("\n" + "=" * 70)
print("MODEL SUMMARY AND FINDINGS")
print("=" * 70)

print("\nModel Architecture:")
model.summary()

print("\nModel Performance:")
print(f"Test Accuracy: {evaluate_history[1] * 100:.2f}%")

print("\nFindings and Insights:")
print("1. The CNN model consists of stacked Conv2D layers with Batch Normalization and MaxPooling.")
print("2. A Dropout layer was added before the Dense layers to reduce overfitting.")
train_acc = train_history.history['sparse_categorical_accuracy'][-1] * 100
val_acc = train_history.history['val_sparse_categorical_accuracy'][-1] * 100
print(f"3. The training accuracy reached {train_acc:.2f}%, while the validation accuracy stabilized at ~{val_acc:.2f}%.")
print("4. This indicates a slightly overfit model, but it still performs well on unseen data.")
test_acc = evaluate_history[1] * 100
print(f"5. The test accuracy was approximately {test_acc:.2f}%, confirming generalization capability.")

print("\nPractical Applications:")
print("This type of CNN-based image classification is useful for:")
print("- Identifying road objects in autonomous vehicles")
print("- Detecting defects in manufacturing pipelines")
print("- Diagnosing plant diseases in agricultural fields")
print("- Enhancing photo sorting/tagging in mobile apps")
print("- Object detection in surveillance and security systems")

# Performance Summary Table
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Metric':<30} {'Value':<15}")
print("-" * 45)
print(f"{'Test Accuracy':<30} {evaluate_history[1]:.4f} ({test_acc:.1f}%)")
print(f"{'Training Accuracy':<30} {train_acc/100:.4f} ({train_acc:.1f}%)")
print(f"{'Validation Accuracy':<30} {val_acc/100:.4f} ({val_acc:.1f}%)")
print(f"{'Test Loss':<30} {evaluate_history[0]:.4f}")
print(f"{'Total Parameters':<30} {model.count_params():,}")
print(f"{'Epochs Trained':<30} {len(train_history.history['loss'])}")
print("=" * 70)

# Save the trained model to disk using the HDF5 format.
model.save('cifar10_cnn_model.h5')
print("\nModel saved as 'cifar10_cnn_model.h5'")
print("\nCIFAR-10 image classification completed!")
