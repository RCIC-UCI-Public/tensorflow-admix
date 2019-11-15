#!/usr/bin/env python

# This code is taken from https://www.tensorflow.org/tutorials/keras/classification
# This guide trains a neural network model to classify images of clothing, like sneakers 
# and shirts. It's okay if you don't understand all the details; this is a fast-paced 
# overview of a complete TensorFlow program with the details explained as you go.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each image is mapped to a single label. Since the class names are not included 
# with the dataset, store them here to use later when plotting the images:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
print ("Explore the data set:")
print ("    training set format:", train_images.shape)
print ("    training set labels size:", len(train_labels))
print ("    training set labels:", train_labels)
print ("    test set format:", test_images.shape)
print ("    test set labels size:", len(test_labels))

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
# To do so, divide the values by 255. It's important that the training set and the testing set
# are preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify that the data is in the correct format and that you're ready to build and train the 
# network, let's display the first 25 images from the training set and display the class name below each image.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

print ("First 25 images")
plt.show()

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Add a few more settings tp the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions: the model has predicted the label for each image 
# in the testing set.
predictions = model.predict(test_images)

# Let's take a look at the first prediction:
print ("First prediction", predictions[0])

# A prediction is an array of 10 numbers. They represent the model's "confidence" that the image 
# corresponds to each of the 10 different articles of clothing. To see which label has the highest confidence value:
print ("Max prediction", np.argmax(predictions[0]))
# So, the model is most confident that this image is an ankle boot, or class_names[9]. 
# Examining the test label shows that this classification is correct:
print ("Prediciton label:", test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# look at 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()

# Finally, use the trained model to make a prediction about a single image.
# Grab an image from the test dataset.
img = test_images[1]
print (img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print (img.shape)

# Now predict the correct label for this image:
predictions_single = model.predict(img)
print (predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# model.predict returns a list of lists—one list for each image in the batch of data. 
# Grab the predictions for our (only) image in the batch:
np.argmax(predictions_single[0])
