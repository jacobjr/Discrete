###################################################
# This file contains imputation techniques. Receiving i,j and k, it will read
# the DB corresponding to these indices, and perform 8 different imputation methods, and write DBs, so they can be classified.
###################################################

from sklearn.preprocessing import Imputer
import pandas as pd
from inoutmd import read
from subprocess import check_output
import numpy as np
import os

folds = 5

# Sklearn imputation methods
methods = ['mean', 'median', 'most_frequent']


def imputation(i, j, k):
    
    # Path creation and reading
    path = str(i) + "-" + str(j) + "-" + str(k) + ".data"
    dataset = read("/Data/" + path)
    # Class not used, in order to get consistent results when classifying
    y = dataset[:, dataset.shape[1]-1]
    dataset = np.delete(dataset, dataset.shape[1]-1,1)

    # For cross validation

    for part in range(0, 5):
        train = dataset[list(map(lambda t: t % 5 != part, range(0, dataset.shape[0]))), :]
        test = dataset[list(map(lambda t: t % 5 == part, range(0, dataset.shape[0]))), :]
        # For each method in SKlearn,
        for w in range(0, 3):

            # Impute
            imp = Imputer(strategy=methods[w], axis=0)
            imp_train = imp.fit_transform(train)
            imp_test = imp.fit_transform(test)

            # Concatenate train and test
            x = np.row_stack((imp_train, imp_test))

            # Concatenate with class
            x = np.column_stack((x, y))

            # Write
            npath = "Data/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(part) + "-" + str(w) + ".data"
            df = pd.DataFrame(x)
            df.to_csv(npath, header=False, index=False)

        # Class in pandas format
        # y = pd.Series(y)

        # Last Value Carried Forward
        # First forward, then backwards, else MV at first rows wont be filled in.
        data = pd.DataFrame(data=train)
        data = data.fillna(method='pad')
        data = data.fillna(method='bfill')

        data1 = pd.DataFrame(data=test)
        data1 = data1.fillna(method='pad')
        data1 = data1.fillna(method='bfill')
        data = np.concatenate((data.as_matrix(), data1.as_matrix()), axis=0)

        data = np.concatenate((data, np.expand_dims(y, axis=1)), axis=1)  # Class concatenate

        # Write
        npath = "Data/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(part) + "-" + str(3) + ".data"
        np.savetxt(npath, data)

        # Interpolation
        data = pd.DataFrame(data=train).astype(float)
        data = data.interpolate(method="polynomial", order=3)

        # Fill possible first rows missing values
        data = data.fillna(method='bfill')
        data = data.fillna(method='pad')

        data1 = pd.DataFrame(data=test).astype(float)
        data1 = data1.interpolate(method="polynomial", order=3)
        data1 = data1.fillna(method='bfill')
        data1 = data1.fillna(method='pad')
        data = pd.concat([data, data1], axis=0)
        data = np.concatenate((data, np.expand_dims(y, axis=1)), axis=1)  # Class

        # Write
        npath = "Data/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(part) + "-" + str(4) + ".data"
        np.savetxt(npath, data)

        # Write for R processes
        npath = "Data/" + str(i) + "-" + str(j) + "-" + str(k) + "-" + str(part) + ".data"
        np.savetxt(npath, dataset)

        # R imputation methods
        a = check_output('Rscript hd.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)
        check_output('Rscript mice.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)
        check_output('Rscript EM.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)

imputation(0, 0, 0)
