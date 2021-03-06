###############################################
# Main file
# Calling: main.py i {i for instances desired}
# Main function makes three calls for three functions.
# each one contains For loops and function calls to other files (inoutmd.py, impute.py, classify.py)
################################################

import argparse
import os

import datetime

from inoutmd import getMDfilesPercent
from classify import classification
from impute import imputation

dbs = ["vehicle.data", "german.data", "bupa.data", "biodeg.data", "foresttypes.data", "diabetic.data", "climate.data",
       "segmentation.data", "thoracic.data", "leaf.data"]
MDT = ["MAR", "MCAR", "MuOV", "MIV"]
methods = ['mean', 'median', 'most_frequent', 'lvcf', 'interpolation', 'HD', 'mice', 'EM']
algorithms = ["Regression L1", "Regression L2", "LDA", "QDA", "C-SVM", "RBFN", "Naive Bayes", "GradientBoost.",
              "RandomForest", "CART", "1-NN", "3-NN"]
# vehicle,german,bupa, biodeg, forest, diabetic, climate, segmentation, thoracic, leaf,
classes = [18, 24, 6, 41, 0, 19, 20, 19, 16,0]  # Classes keeps the column index where the class can be found in each DB.

percentages = [2,3,4,5,6]
# For each DB in dbs array, a DB is read and introduced MD

def generateMDfiles(instances):
    """

    :param instances: Number of times the experiment will be run
    :return: For each instance, DB, and MDT combination, this function writes sends a process that will produce 5
     incomplete databases in files, that will posteriorly be imputed and used for classification.
    """
    print("Generating MD files started at " + str(datetime.datetime.now()))

    if not os.path.exists("Data"):  # Path where files will be saved
        os.makedirs("Data")

    # Files with MD are generated with xyz.data names, representing x <- database, y <- MDtype and z<-instance
    for i in [0, 9]: #range(0, len(dbs)):
        for j in range(0, len(percentages)):
            for k in range(0, instances):
                # In this function MD is introduced apart from reading (watch inoutmd.py)
                getMDfilesPercent(dbs[i], i, percentages[j], k, classes[i], 1, k*100+j*10+i)
    print("And finished at " + str(datetime.datetime.now()))


def generateImputeds(instances):
    """

    :param instances: Number of times the experiment will be run
    :return: For each combination generated by the function above, this function will write 8 files, that will
    contain the imputed versions of the original file (each one imputed by one imputer)
    """
    # Previous files are filled in, xyzn.data names, continuing xyz the same, and being n <- imputation method
    print("Imputing MD files started at " + str(datetime.datetime.now()))
    for i in [0, 9]: #range(0, len(dbs)):
        for j in range(0, len(percentages)):
            for k in range(0, instances):
                imputation(i, percentages[j], k)  # see (impute.py)

    print("And finished at " + str(datetime.datetime.now()))

    # Accuracies are calculated for imputed files.
    # These will be saved in Output.csv.


def generateAccuracies(instances, folds, i0, j0, k0, n0):
    """

    :param instances: Number of times the experiment will be run
    :param folds: folds to be used in classification
    :param i0: Initialization parameter to start executions from a given point, if stopped.
    :param j0: Initialization parameter to start executions from a given point, if stopped.
    :param k0: Initialization parameter to start executions from a given point, if stopped.
    :param n0: Initialization parameter to start executions from a given point, if stopped.
    :return: Analogously, this function considers all imputed files and generates accuracies from them
    """
    print("Classifying imputed files started at " + str(datetime.datetime.now()))

    # This makes it possible to continue an execution in case it needed to be stopped (see main function).
    for i in [0,9]: #range(2, len(dbs)):
        for j in range(0, len(percentages)):
            for k in range(0, 30):
                for n in range(0, len(methods)):
                    classification(i, percentages[j], k, n, [0,2,9,10], folds)  # Classification algorithms execution (see classify.py)
    print("And finished at " + str(datetime.datetime.now()))


if __name__ == '__main__':
    """parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(50),
        nargs='+', help='number of generated instances for each DB/MDtype')

    args = parser.parse_args()
    """
    instances = 30  # Instances for each BD with the same MD type

    generateMDfiles(instances)

    generateImputeds(instances)
    folds = 5

    # Four parameters as start point. To start from the beginning, four 0s, as above.
    # in case that last execution finished at (0,1,2,3,4) continuing point is (0,1,2,3,5)
    # generateAccuracies(instances, 3, 0, 0, 0) This will start in the 4th database
    generateAccuracies(instances, folds, 0, 0, 0, 0)  # This will start from the beginning


