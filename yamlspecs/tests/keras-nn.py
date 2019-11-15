#!/usr/bin/env python

# This code is taken from https://www.tensorflow.org/tutorials/quickstart/beginner
# This short introduction uses Keras to:
#    Build a neural network that classifies images.
#    Train this neural network.
#    And, finally, evaluate the accuracy of the model.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf

# Load and prepare the MNIST dataset. 
# Convert the samples from integers to floating-point numbers:
print ("### Load and prepare the MNIST dataset")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers. 
# Choose an optimizer and loss function for training:
print ("### Building tf.keras.Sequential model by stacking layers")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train and evaluate the model:
model.fit(x_train, y_train, epochs=5)
print ("### Training model")
model.evaluate(x_test,  y_test, verbose=2)
print ("### Evaluating model")
