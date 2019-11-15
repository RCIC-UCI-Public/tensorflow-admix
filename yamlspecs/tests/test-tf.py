#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops as _ops
from tensorflow.python.platform import test as _test
from tensorflow.python.platform.test import *

import tensorflow as tf

#print ("Contents of tensorflow.python.framework: ")
#print (dir(_ops))

#tf.enable_eager_execution() - error this does not exist
_ops.enable_eager_execution()

ans = tf.add(1, 2).numpy()
print ("1 + 2 =", ans)

hello = tf.constant('Hello, TensorFlow!')
print (hello.numpy())
