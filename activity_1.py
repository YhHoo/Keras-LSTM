import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# import MINST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders to hold training data
# input x - for 28 x 28 pixels = 784 nodes
x = tf.placeholder(tf.float32, [None, 784])
# output y - for 1-10 digits = 10 nodes
y = tf.placeholder(tf.float32, [None, 10])

# Input nodes = 784, Hidden nodes = 300, Output nodes = 10
# Declare weights n bias fr Input to Hidden
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# Declare weights n bias fr Hidden to Output
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# Calc output of network
# Hidden Layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
# Output Layer
y_prime = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# Cost function
# clip the y_prime output from NN to 1e-10 n 0.99
y_prime_clipped = tf.clip_by_value(y_prime, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_prime_clipped) +
                                              (1 - y) * tf.log(1 - y_prime_clipped), axis=1))

# Add optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# setup initialization operator
init_op = tf.global_variables_initializer()

# define accuracy assessment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prime_clipped, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:

    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


