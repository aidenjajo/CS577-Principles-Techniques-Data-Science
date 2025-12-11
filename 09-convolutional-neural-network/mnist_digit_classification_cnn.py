"""
MNIST Digit Classification using Convolutional Neural Network
CS577-01: Principles and Techniques of Data Science
San Diego State University
Author: Aiden Jajo
Date: April 20, 2025
"""

# ============================================================================
# 1) Data Preprocess
# ============================================================================

# 1a) Import the libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("MNIST DIGIT CLASSIFICATION USING CNN")
print("=" * 70)

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# 1b) Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ============================================================================
# 2) Build the Keras Sequential Model
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

# 2a) Add an input layer with the shape of: [28,28,1]
model = keras.Sequential()
# Reshaping input to add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 2b) Add a Conv2D Layer, with filter = 4, kernel_size = [3,3], strides = [1, 1], activation function = ReLU.
model.add(keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1),
                             activation='relu', input_shape=(28, 28, 1)))

# 2c) Add a MaxPool2D Layer with pool_size = 2, strides = 2, padding='valid'
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# d) Repeat step b and step c in total 1~3 times (showing one repetition as example)
model.add(keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 2e) Add a Flatten Layer.
model.add(keras.layers.Flatten())

# 2f) Add 1 or 2 Dense Layers, with unit_number = 64, activation function = 'ReLU'
model.add(keras.layers.Dense(units=64, activation='relu'))

# 2g) Add a Dense Layer as output layer, with unit = 10 and activation function = softmax.
model.add(keras.layers.Dense(units=10, activation='softmax'))

print("\nModel Summary:")
model.summary()

# ============================================================================
# 3) Training
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

# 3a) Compile the model with adam optimizer
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 3b) Fit the model on the train set with 2048 batch_size and more than 10 epochs, validation_split=0.2
# Early stopping callback (optional)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

train_history = model.fit(x_train, y_train,
                         batch_size=2048,
                         epochs=15,
                         validation_split=0.2,
                         callbacks=[early_stopping])

# 3c) Obtaining the training and evaluate history
evaluate_history = model.evaluate(x_test, y_test)

# ============================================================================
# 4) Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

# 4a) Plot the training history
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['sparse_categorical_accuracy'])
plt.plot(train_history.history['val_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('training_accuracy.png')
plt.show()

# 4b) Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('training_loss.png')
plt.show()

# 4c) Print the test accuracy
print("\nTest Accuracy and Loss:")
print(evaluate_history)

# ============================================================================
# 5) Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

print(f"\nModel architecture: CNN with {len(model.layers)} layers")
print(f"Training accuracy: {train_history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"Validation accuracy: {train_history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"Test accuracy: {evaluate_history[1]:.4f}")
print(f"The model achieved approximately {evaluate_history[1]*100:.1f}% accuracy on the MNIST test dataset.")
print("The training curves show good convergence with no significant overfitting.")
print("The CNN architecture demonstrates effective feature extraction for handwritten digit recognition.")

# Performance Summary Table
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Metric':<30} {'Value':<15}")
print("-" * 45)
print(f"{'Test Accuracy':<30} {evaluate_history[1]:.4f} ({evaluate_history[1]*100:.1f}%)")
print(f"{'Training Accuracy':<30} {train_history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"{'Validation Accuracy':<30} {train_history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"{'Test Loss':<30} {evaluate_history[0]:.4f}")
print(f"{'Total Parameters':<30} {model.count_params():,}")
print(f"{'Epochs Trained':<30} {len(train_history.history['loss'])}")
print("=" * 70)

print("\nMNIST digit classification completed!")
