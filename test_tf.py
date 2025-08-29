import tensorflow as tf
import numpy as np

print("Testing TensorFlow installation...")
print(f"TensorFlow version: {tf.__version__}")

# Check if GPU is available
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Simple test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

c = tf.matmul(a, b)
print("Matrix multiplication test:")
print(c)

print("TensorFlow is working correctly!")