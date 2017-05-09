###############################################
# Main file
# Calling: main.py f {f for folds in classification}
# Main function makes three calls for three functions.
# each one contains For loops and function calls to other files (inoutmd.py, impute.py, classify.py)
################################################

import argparse
import os

import datetime

from inoutmd import getMDfiles
from classify import classification
from impute import imputation

md = ["MCAR", "MAR", "MuOV", "MIV"]
dbs = ["vehicle.data", "german.data", "bupa.data", "biodeg.data", "foresttypes.data", "diabetic.data", "climate.data",
       "segmentation.data", "Thoracic.data", "leaf.data"]
MDT = ["MAR", "MCAR", "MuOV", "MIV"]
methods = ['mean', 'median', 'most_frequent', 'lvcf', 'interpolation', 'HD', 'mice', 'EM']
algorithms = ["Regression L1", "Regression L2", "LDA", "QDA", "C-SVM", "RBFN", "Naive Bayes", "GradientBoost.",
              "RandomForest", "CART", "1-NN", "3-NN"]
# vehicle,german,bupa, biodeg, forest, diabetic, climate, segmentation, thoracic, leaf,
classes = [18, 24, 6, 41, 0, 19, 20, 0, 16,
           0]  # Classes guarda las columnas en las que la clase se encuentra en cada DB


# For each DB in dbs array, a DB is read and introduced MD


def generateMDfiles(instances):
    print("Generating MD files started at " + str(datetime.datetime.now()))

    if not os.path.exists("Data"):  # Path where files will be saved
        os.makedirs("Data")

    # Files with MD are generated with xyz.data names, representing x <- database, y <- MDtype and z<-instance
    for i in range(0, len(dbs)):
        for j in range(0, len(MDT)):
            for k in range(0, instances):
                # In this function MD is introduced apart from reading (watch inoutmd.py)
                getMDfiles(dbs[i], i, j, k, classes[i], 1)
    print("And finished at " + str(datetime.datetime.now()))


def generateImputeds(instances):
    # Previous files are filled in, xyzn.data names, continuing xyz the same, and being n <- imputation method
    print("Imputing MD files started at " + str(datetime.datetime.now()))
    for i in range(0, len(dbs)):
        for j in range(0, len(MDT)):
            for k in range(0, instances):
                imputation(i, j, k)  # see (impute.py)

    print("And finished at " + str(datetime.datetime.now()))

    # Accuracies are calculated for imputed files.
    # These will be saved in Output.csv.


def generateAccuracies(instances, folds, i0, j0, k0, n0):
    print("Classifying imputed files started at " + str(datetime.datetime.now()))

    # This makes it possible to continue an execution in case it needed to be stopped (see main function).
    if i0 == 0 and j0 == 0 and k0 == 0 and n0 == 0:  # If we start from the beginning, erase content in Output.csv
        text_file = open("Output.csv", "w")
    else:
        text_file = open("Output.csv", "a")  # else, append
    act = False
    for i in range(0, len(dbs)):
        for j in range(0, len(MDT)):
            for k in range(0, instances):
                for n in range(0, len(methods)):
                    if i == i0 and j == j0 and k == k0 and n == n0:
                        act = True
                    if act:
                        path = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(
                            n) + ".data"  # Path creation for imputed file
                        acc = classification(i, j, k, n, range(0, len(algorithms)),
                                             folds)  # Classification algorithms execution (see classify.py)
                        for m in range(0, len(algorithms)):
                            # Write in Output.csv
                            text_file.write(
                                str(i) + "," + str(j) + "," + str(k) + "," + str(n) + "," + str(m) + "," + str(
                                    acc[m]) + "\n")
                            # ########These three lines are for being able to observe progress
                            text_file.flush()
                            os.fsync(text_file)

                            #############################################################
    text_file.close()
    print("And finished at " + str(datetime.datetime.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(50),
        nargs='+', help='number of generated instances for each DB/MDtype')

    args = parser.parse_args()
    instances = args.integers[0]  # Instances for each BD with the same MD type

    # generateMDfiles(instances)

    # generateImputeds(instances)
    folds = 5

    # Four parameters as start point. To start from the beginning, four 0s, as above.
    # in case that last execution finished at (0,1,2,3,4) continuing point is (0,1,2,3,5)
    # generateAccuracies(instances, 3, 0, 0, 0) This will start in the 4th database
    generateAccuracies(instances, folds, 8, 0, 0, 0)  # This will start from the beginning
