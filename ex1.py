#!/usr/bin/python

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.1
training_epochs = 100

# Network Parameters
n_input = 1
n_hidden_1 = 5
n_out = 1

x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)

print inp
print tar

x = tf.constant(inp)
y = tf.constant(tar)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
h3 = tf.Variable(tf.random_normal([n_hidden_1, n_out]))

b1 = tf.Variable(tf.random_normal([n_hidden_1]))
b3 = tf.Variable(tf.random_normal([n_out]))

layer_1 = tf.add(tf.matmul(x, h1), b1)
layer_1 = tf.sigmoid(layer_1)

# Hidden layer with RELU activation
#layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
#layer_2 = tf.nn.relu(layer_2)

# Output layer with linear activation
out_layer = tf.matmul(layer_1, h3)
#out_layer = tf.sigmoid(out_layer)

cost = tf.reduce_mean(tf.square(out_layer - y))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: inp,
                                                      y: tar})
        print c

