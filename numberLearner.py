import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io as scio
import featureNormalize
import sigmoid
import logisticRegression


dataset = scio.loadmat('./machine-learning-ex3/ex3/ex3data1.mat')



x_train = dataset['X']
m,n = x_train.shape
x_train = np.concatenate((np.ones((m,1)), x_train), axis=1)
print(x_train.shape)
y_train = dataset['y']
theta = np.zeros((n+1,1))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
theta_var = tf.Variable(theta, dtype=tf.float32, name='Weight')


sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

h = sigmoid.sigmoid(tf.matmul(x, theta_var))
theta_trained = theta

for i in range(1, 11):
    theta_trained = logisticRegression.logisticTrainer(theta_trained, x_train, y_train==i, 0.1, 0.01)


print(np.matmul(x_train, theta_trained)-y_train)






