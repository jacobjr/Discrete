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
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Import MNIST data
#warnings.filterwarnings("ignore")
#tf.logging.set_verbosity(tf.logging.ERROR)

LOGDIR = "/tmp/multi_label/"
data = np.loadtxt("emotions.dat", delimiter=",")
mean_acc = 0
fold = 0
for x, y in KFold(n_splits=5).split(data[:,-6:], data[:,:-6]):
    train_labels = data[x,-6:]
    train_features = data[x,:-6]
    test_labels = data[y,-6:]
    test_features = data[y,:-6]
    # Parameters
    learning_rate = 0.01
    training_epochs = 1000
    batch_size = 100
    display_step = 1000

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 72]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 6]) # 0-9 digits recognition => 10 classes

    w = tf.Variable(tf.truncated_normal([72, 6]))

    b = tf.Variable(tf.truncated_normal([6]))

    logit = tf.matmul(x, w) + b

    tf.summary.histogram("Weights", w)
    tf.summary.histogram("Biases", b)
    #tf.summary.histogram("Logit", logit)

    aux = tf.map_fn(lambda arg: tf.map_fn(lambda arg1: tf.cond(tf.less(0.3, arg1), lambda: tf.constant(1, dtype=tf.float32, name='one'), lambda: tf.constant(0, dtype=tf.float32, name='one')), arg), logit)

    hamming_loss = tf.divide(tf.divide(tf.reduce_sum(tf.abs(tf.subtract(aux, y))), tf.cast(tf.shape(y)[0], tf.float32)), tf.cast(tf.shape(y)[0], tf.float32))

    #tf.summary.scalar("Hamming", hamming_loss)

    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

    #tf.summary.scalar("XEntropy", cost)

    # Gradient Descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    summ = tf.summary.merge_all()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(LOGDIR + "Fold" + str(fold))
        writer.add_graph(sess.graph)
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_features.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_features[i*batch_size:(1+i)*batch_size,:]
                batch_ys = train_labels[i*batch_size:(1+i)*batch_size]
                #print(batch_ys, batch_xs)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                #print(c)
                avg_cost += c / total_batch
                if i % 50 == 0:
                    #print(batch_xs.shape, batch_ys.shape)
                    s = sess.run(summ, feed_dict={x: batch_xs, y: batch_ys})
                    writer.add_summary(s, epoch)
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        mean_acc += accuracy.eval({x: test_features, y: test_labels})/5
        print("Hamming loss: ", hamming_loss.eval({x: test_features, y: test_labels}))
        fold += 1
        #print(aux.eval({x: test_features, y: test_labels}))
        #print(test_labels)
