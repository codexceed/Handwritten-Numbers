import tensorflow as tf
import numpy as np
import sigmoid


def logisticTrainer(theta_init, X, Y, lamb, alpha):

    m,n = X.shape

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    theta = tf.Variable(theta_init, dtype=tf.float32, name='weights')

    h = sigmoid.sigmoid(tf.matmul(x, theta))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    theta_mod = theta_init


    J = (-1/m)*tf.reduce_sum((y*tf.log(h)+(1-y)*tf.log(1-h)), 0) + (lamb/(2*m))*(tf.reduce_sum(theta**2) - sess.run(theta)[0]**2)

    optimizer = tf.train.GradientDescentOptimizer(alpha)

    train = optimizer.minimize(J)

    for i in range(100):
        sess.run(train, {x:X, y:Y})

    print(sess.run(J, {x:X, y:Y}))


    return sess.run(theta)
