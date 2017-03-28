import numpy as np
import tensorflow as tf

def sigmoid(X):
    g = tf.exp(-X)
    return (1/(1+g))