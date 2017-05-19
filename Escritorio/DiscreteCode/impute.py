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

    """
    This function takes as input three indices, which correspond to a single incomplete file (previously generated
    in inoutmd.getMDfiles(). Reads the file and imputes it with different IMs. Then writes the results with the
    same indices, but havind added one, the IM index. For example, the "0-0-0-1.data" will result in "0-0-0-1-x.data"
    with x ranging from 0 to 7.


    :param i: Database index
    :param j: MDT index
    :param k: Instance index
    """
    f = open("ImpLog.txt", "a+")

    for part in range(0, 5):
        # Path creation and reading
        path = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(part) + ".data"
        dataset = read("/Data/" + path)
        # Class not used, in order to get consistent results when classifying
        y = dataset[:, dataset.shape[1]-1]
        dataset = np.delete(dataset, dataset.shape[1]-1,1)

        # For cross validation
        train = dataset[:int(dataset.shape[0]/5*4), :]
        test = dataset[int(dataset.shape[0]/5*4):, :]

        # For each method in SKlearn,
        for w in range(0, 3):
            try:
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
                np.savetxt(npath, x, delimiter=",")
            except:
                f.write(methods[w] + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + "\n")

        # Class in pandas format

        # Last Value Carried Forward
        # First forward, then backwards, else MV at first rows wont be filled in.
        try:
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
            np.savetxt(npath, data, delimiter=",")
        except:
            f.write("LVCF " + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + "\n")

        # Interpolation
        try:
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
            np.savetxt(npath, data, delimiter=",")
        except:
            f.write('poly {0} {1} {2} {3}\n'.format(str(i), str(j), str(k), str(part)))

        # R imputation methods
        try:
            check_output('Rscript hd.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)
        except:
            f.write('hd ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + "\n")
        try:
            check_output('Rscript mice.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)
        except:
            f.write('mice ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + "\n")
        try:
            check_output('Rscript EM.r ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + " " + str(train.shape[0]) + " " + os.path.dirname(os.path.abspath(__file__)) + "/Data/", shell=True)
        except:
            f.write('EM ' + str(i) + " " + str(j) + " " + str(k) + " " + str(part) + "\n")

    f.close()
#imputation(0, 0, 0)
