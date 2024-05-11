#!/usr/bin/env python

from tensorflow import keras
import numpy as np
t = np.ones([5,32,32,3])

c1 = keras.layers.Conv2D(32, 3, activation="relu")
print ("Using keras.layers.Conv2D", c1(t))

c2 = keras.layers.Dense(32, activation="relu")
print ("Using keras.layers.Dense", c2(t))
