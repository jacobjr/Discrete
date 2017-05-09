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

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from inoutmd import read
import numpy as np
from random import choice

# Principal classification function, given a path where the DB is, an array of classifiers, and an amount of folds, it returns
# an array of accuracies, with length = length(classifiers)


def classification(i, j, k, n, classifiers, folds):
    res = []

    for c in classifiers:  # Foreach classifier,
        res.append(classify(i, j, k, n, c, folds))  # Classify
    return res


def classify(i, j, k, n, c, folds):
    
    # ##############Selected classifier ################# #
    if c == 0:
        clf = LogisticRegression(penalty="l1")
    elif c == 1:
        clf = LogisticRegression(penalty="l2")
    elif c == 2:
        clf = LinearDiscriminantAnalysis(solver="lsqr")
    elif c == 3:
        clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    elif c == 4:
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd", learning_rate="adaptive")
    elif c == 5:
        clf = SVC(kernel='linear', C=1.0, tol=0.001, gamma=0.01, coef0=0, probability=True)
    elif c == 6:
        clf = SVC(kernel='rbf', C=1.0, gamma=0.10000000000000001, coef0=0, shrinking=True, probability=True)#RBFN
    elif c == 7:
        clf = GaussianNB() 
    elif c == 8:
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=11, subsample=1.0)
    elif c == 9:
        clf = RandomForestClassifier(n_estimators=10)
    elif c == 10:
        clf = tree.DecisionTreeClassifier()  # CART, similar to c4.5
    elif c == 11:
        clf = KNeighborsClassifier(n_neighbors=1)
    elif c == 12:
        clf = KNeighborsClassifier(n_neighbors=3)
    #######################################################################

    for fold in range(0, folds):
        path = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(fold) + "-" + str(n) + ".data"
        x = read(path)

        # Separate class (always in last position, watch impute.py)
        y = x[:, len(x[0, :])-1]
        x = np.delete(x, len(x[0, :])-1, 1)

        lim = x.shape[0] / folds * (folds - 1)
        x_train = x[:lim,:]
        x_test = x[lim:,:]
        y_train = y[:lim]
        y_test = y[lim:]

        # If class is string, transform to numeric labels
        if isinstance(y[0], basestring):
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)

        model = clf.fit(x_train, y_train)  # Model creation
        predictions = model.predict(x_test)
        
    # Classification
    for i, (train, test) in enumerate(cv):

        xaux = x[train]
        taux = x[test]
                
        model = clf.fit(xaux, y[train])  # Model creation
        probas = model.predict_proba(taux)  # Classify

        if np.isnan(probas[:, 1]).any():
            print("Classifier error:", c)

        # We put in res the class with the most probability
        res = []
        
        # Class with most probability (same probability problem)
        for k in range(0,probas.shape[0]):
            max = 0
            maxi = []
            for j in range(0,probas.shape[1]):
                
                # If more than one class has the same probability, random selection between them
                if probas[k, j] > max:  # If the jth probability is larger than the max observed, actualize max, and reinitialize array maxi
                    max = probas[k, j]
                    maxi = [j]
                elif probas[k, j] == max:  # If the jth probability matches the max observed, add j to the array
                    maxi.append(j)
            if len(maxi) == 0:  # If len is 0, an error happened, -1 as non existing class
                res.append(-1)
            else:  # Else, correct classifying, and random choice between all the max probabilities
                res.append(choice(maxi))
       
        # Accuracy for one single fold
        acc = accuracy_score(y[test], res)
        # Accumulated accuracy
        acc1 += acc*len(test)/(len(train)+len(test))
    return acc1
