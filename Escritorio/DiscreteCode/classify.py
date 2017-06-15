##################################################
# This file contains classification algorithms.
# Callable function is classification, which returns an array of accuracies corresponding to the classifiers array.
##################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import os
import sys
import warnings
from sklearn import preprocessing
from inoutmd import read
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)


# Principal classification function, given a path where the DB is, an array of classifiers, and an amount of folds, it returns
# an array of accuracies, with length = length(classifiers)


def classification(i, j, k, n, classifiers, folds):
    """
    This function takes as input path indices and a list of classifier indices and returns a list of
    accuracies generated from the file represented by the indices and each one of the classifiers.

    :param i: DB index
    :param j: MDT index
    :param k: Instance index
    :param n: IM index
    :param classifiers: list of classifier indices (usually range(0, 15))
    :param folds: Number of folds. Fixed beforehand.
    :return: List of accuracies generated from the different combinations of the previous parameters
    """
    res = None
    try:
        for c in classifiers:  # Foreach classifier,
            if not res:
                res = classify(i, j, k, n, c, folds)
            else:
                res.append(classify(i, j, k, n, c, folds)[1])  # Classify
        res = np.array(res)
        np.savetxt("ClassificationsMD/{0}-{1}-{2}-{3}.data".format(str(i), str(j), str(k), str(n)), res, fmt='%i')
    except:
        print("{0}-{1}-{2}-{3}.data".format(str(i), str(j), str(k), str(n)))

    return res


def classify(i, j, k, n, c, folds):
    """
    This function reads the file represented by i, j, k, and n indices and generates an accuracy using "folds" folds
    and the classifier indexed by c.
    :param i: DB index
    :param j: MDT index
    :param k: Instance index
    :param n: IM index
    :param c: classifier index
    :param folds: Number of folds. Fixed beforehand.
    :return: Accuracy generated from the use of classifier c on the data described in the file represented by the
    indices.
    """

    # ##############Select classifier ################# #
    if c == 0:
        clf = LogisticRegression(penalty="l1")
    elif c == 1:
        clf = LogisticRegression(penalty="l2")
    elif c == 2:
        clf = LinearDiscriminantAnalysis(solver="lsqr")
    elif c == 3:
        clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    elif c == 4:
        x = read("DataMD/{0}-{1}-{2}-{3}.data".format(str(i), str(j), str(k), "0"), delimiter=",")
        size = x.shape[0]
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1]-1)]
        clf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[int(size/3*2), int(size/3)], n_classes=len(set(x[:, -1]))+1)
    #elif c == 5:
        #clf = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10])
    elif c == 6-1:
        clf = SVC(kernel='linear', C=1.0, tol=0.001, probability=True)
    elif c == 7-1:
        clf = SVC(kernel='poly', C=1.0, tol=0.01, probability=False, degree=2, cache_size=20000)
    elif c == 8-1:
        clf = SVC(kernel='rbf', C=1.0, gamma=0.10000000000000001, coef0=0, shrinking=True, probability=True)#RBFN
    elif c == 9-1:
        clf = GaussianNB() 
    elif c == 10-1:
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=11, subsample=1.0)
    elif c == 11-1:
        clf = RandomForestClassifier(n_estimators=10)
    elif c == 12-1:
        clf = tree.DecisionTreeClassifier()  # CART, similar to c4.5
    elif c == 13-1:
        clf = KNeighborsClassifier(n_neighbors=1)
    elif c == 14-1:
        clf = KNeighborsClassifier(n_neighbors=3)
    #######################################################################

    full_predictions = []
    full_y = []

    for fold in range(0, folds):
        # For each stratified fold, we have one file. We read it.
        path = "DataMD/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(fold) + "-" + str(n) + ".data"
        x = read(path)
        x = preprocessing.Imputer().fit_transform(x)
        x[x == np.nan] = 0
        # Separate class (always in last position, watch impute.py)
        y = x[:, len(x[0, :])-1]
        x = np.delete(x, len(x[0, :])-1, 1)

        # If class is string, transform to numeric labels
        if isinstance(y[0], str):
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        # Set where the limit between the train and testing is. Remember that we always write first the training
        # part and then the testing part.

        lim = int(x.shape[0] / folds * (folds - 1))
        x_train = x[:lim,:]
        x_test = x[lim:,:]
        y_train = y[:lim].astype(int)
        y_test = y[lim:].astype(int)

        def get_train_inputs():
            x = tf.constant(x_train)
            y = tf.constant(y_train)

            return x, y

        def get_test_inputs():
            x = tf.constant(x_test)
            y = tf.constant(y_test)

            return x, y

        def get_predict_input():
            x = tf.constant(x_test)

            return x

        if c == 4:
            model = clf.fit(input_fn=get_train_inputs, max_steps=20000)  # Model creation
            predictions = model.predict(input_fn=get_predict_input)
            # acc = clf.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
        else:
            model = clf.fit(x_train, y_train)  # Model creation
            predictions = model.predict(x_test)

        full_predictions += list(predictions)
        full_y += y_test.tolist()
        #print(full_predictions)
    # print(accuracy_score(full_y, full_predictions))
    return [full_y, full_predictions]


def nn(x_train, y_train, x_test, y_test):

    """
    A Convolutional Network implementation example using TensorFlow library.
    This example is using the MNIST database of handwritten digits
    (http://yann.lecun.com/exdb/mnist/)

    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    """

    # Parameters
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 128

    # Network Parameters
    n_input = x_train.shape[1]
    n_classes = len(set(y_train)) # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(x, weights, biases, dropout):


        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, n outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x = x[((step-1) * batch_size):(step * batch_size),:], batch_y = y[((step-1) * batch_size):(step * batch_size)]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        return "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: x_test,
                                          y: y_test,
                                          keep_prob: 1.})

"""
path = "Data/" + str(0) + "-" + str(0) + "-" + str(0) + "-" + str(0) + "-" + str(0) + ".data"
x = read(path)
x = preprocessing.Imputer().fit_transform(x)
# Separate class (always in last position, watch impute.py)
y = x[:, len(x[0, :])-1]
x = np.delete(x, len(x[0, :])-1, 1)

a = nn(x[:int(x.shape[0]/5*4),:], y[:int(y.shape[0]/5*4)], x[int(x.shape[0]/5*4):,:], y[int(y.shape[0]/5*4):])
print(a)
"""
