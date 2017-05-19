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
import datetime
import os
import warnings
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
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

    for c in classifiers:  # Foreach classifier,
        if not res:
            res = classify(i, j, k, n, c, folds)
        else:
            res.append(classify(i, j, k, n, c, folds)[1])  # Classify
    res = np.array(res)
    np.savetxt("Classifications/{0}-{1}-{2}-{3}.data".format(str(i), str(j), str(k), str(n)), res, fmt='%i')

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
        x = read("Data/{0}-{1}-{2}-{3}.data".format(str(i), str(j), str(k), "0"), delimiter=",")
        size = x.shape[0]
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1]-1)]
        clf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[size, int(size/2), int(size/2)], n_classes=len(set(x[:, -1])))
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
        path = "Data/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(fold) + "-" + str(n) + ".data"
        x = read(path)

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
            model = clf.fit(input_fn=get_train_inputs, steps=20000)  # Model creation
            predictions = model.predict(input_fn=get_predict_input)
            # acc = clf.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
        else:
            model = clf.fit(x_train, y_train)  # Model creation
            predictions = model.predict(x_test)

        full_predictions += list(predictions)
        full_y += y_test.tolist()
    #print(accuracy_score(full_y, full_predictions))
    return [full_y, full_predictions]
#a = classification(0, 0, 0, 5, range(0, 6), 5)
#print(a)
