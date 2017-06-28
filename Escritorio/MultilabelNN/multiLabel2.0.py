'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import warnings
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Import MNIST data
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

def batch(x, y, size, i):
    """

    :param x: 2darray containing all rows and features in the data
    :param y: 2darray with the corresponding labels
    :param size: size of the batch to be given
    :param i: last index used
    :return: new last index used, x_batch and y_batch of size size.
    """
    if i + size > x.shape[0]:
        index = i + size-x.shape[0]
        return index, np.concatenate((x[i:,:], x[:index,:])), np.concatenate((y[i:], y[:index]))
    else:
        index = i+size
        return index, x[i:index,:], y[i:index]

LOGDIR = "Board/" # Folder where log is stored

# Remove previous log.
if os.path.exists(LOGDIR):
    shutil.rmtree(LOGDIR)
#
data = np.loadtxt("emotions.dat", delimiter=",") # Load data


# Initialize tensors
x = tf.placeholder(tf.float32, [None, 72], name="x") # x will contain the data batches
y = tf.placeholder(tf.float32, [None, 6], name ="y") # y will contain the label batches

w = tf.Variable(tf.truncated_normal([72, 6]), name="w") # w will contain the weights

b = tf.Variable(tf.truncated_normal([6]), name="b") # b will contain the biases

logit = tf.matmul(x, w) + b # logit function

aux = tf.map_fn(lambda arg: tf.map_fn(lambda arg1: tf.cond(tf.less(0.3, arg1), lambda: tf.constant(1, dtype=tf.float32, name='one'), lambda: tf.constant(0, dtype=tf.float32, name='one')), arg), logit)

hamming_loss = tf.divide(tf.divide(tf.reduce_sum(tf.abs(tf.subtract(aux, y))), tf.cast(tf.shape(y)[0], tf.float32)), tf.cast(tf.shape(y)[0], tf.float32))

tf.summary.scalar("Hamming", hamming_loss)

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

# Values to be logged

tf.summary.histogram("Weights", w)
tf.summary.histogram("Biases", b)
tf.summary.histogram("Logit", logit)
tf.summary.scalar("XEntropy", cost)

# Gradient Descent
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
summ = tf.summary.merge_all()
fold = 0
index = 0
# Launch the graph
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

learning_rate = 0.01
training_epochs = 20000
batch_size = 100
display_step = 100

saver = tf.train.Saver()

for x1, y1 in KFold(n_splits=5).split(data[:,-6:], data[:,:-6]):

    train_labels = data[x1,-6:]
    train_features = data[x1,:-6]
    test_labels = data[y1,-6:]
    test_features = data[y1,:-6]

    if fold == 0:
        writer = tf.summary.FileWriter(LOGDIR + "Graph")
        writer.add_graph(sess.graph)

    writer = tf.summary.FileWriter(LOGDIR + "Fold" + str(fold))
    for epoch in range(training_epochs):
        avg_cost = 0.

        index, batch_xs, batch_ys = batch(train_features, train_labels, batch_size, index)
        _, c, weights, bias = sess.run([optimizer, cost, w, b], feed_dict={x: batch_xs, y: batch_ys})

        avg_cost += c / batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % epoch, "cost =", "{:.9f}".format(avg_cost))

            s = sess.run(summ, feed_dict={x: batch_xs, y: batch_ys, w: weights, b: bias})
            writer.add_summary(s, epoch)
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), epoch)
    fold += 1

    print("Optimization Finished!")

    print("Hamming loss: ", hamming_loss.eval({x: test_features, y: test_labels}, session=sess))

