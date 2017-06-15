############################################################
# This file contains two function types:
# First four get a database in table format (without the class attribute)
# and introduce a different kind of MD into it. Finally, they return it with
# some of its values as "NaN"

# Last two are input/output functions.
############################################################
import numpy as np
import pandas as pd
import random
import copy
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

percent = 7  # This work considers introducing 7% of missing values in MD free DBs
nVar = 3

# ####################### Missing Data inserting functions ###################


# This function introduces MCAR type MD.
def insertMCAR(data, per):
    x = data.shape[0]
    y = data.shape[1]
    for i in range(0, x*y*percent//100):
        data[random.randint(0,x-1), random.randint(0,y-1)] = "NaN"  # It randomly selects two indices from the table and assign a "NaN"
         
    return data


# This function introduces MuOV type MD
def insertMuOV(data, per):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), nVar*per)  # Select the features losing their values
    while len(obs) < (percent*x*y/100)/nVar*per:
        obs.append(random.randint(0, x-1))  # Since the "causative" is unobserved, the observations missing their values are selected randomly
    for i in range(0, len(obs)):  # For the selected observations (above)
        for j in range(0, len(y1)):  # For the selected features
            data[obs[i], y1[j]] = "NaN"  # "NaN" assigning

    return data


# This function introduces MIV type MD
def insertMIV(data, per):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), nVar*per)  # Select which features will lose values
    for i in range(0, len(y1)):

        # Auxiliary variable, to select the observations losing values, without modifying them yet.
        auxy = copy.copy(data[:, y1[i]])
        while len(obs) < (percent*y*x/100)/nVar*per:
            obs.append(np.argmin(auxy))
            auxy[obs[len(obs)-1]] = 999999

        for j in range(0, len(obs)):
            data[obs[j], y1[i]] = "NaN"  # "NaN" assigning for these positions
        obs = []
    return data


# This function introduces MAR type MD
def insertMAR(data, per):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), nVar*per+1)  # First element in y1 will be the "causative" variable, remaining three will lose values

    # Auxiliary causative variable, to select the observations losing values, without modifying the causative.
    auxy = copy.copy(data[:, y1[0]])
    while len(obs) < (percent*x*y/100)/nVar*per:
        obs.append(np.argmin(auxy))
        auxy[obs[len(obs)-1]] = 999999

    for i in range(0, len(obs)):
        for j in range(1, len(y1)):
            data[obs[i], y1[j]] = "NaN"  # "NaN" assigning

    return data


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ####################### In/Out functions ######################################

# Simply read. Used to read imputed files.
def read(path, delimiter=","):

    """
    This function takes a path as an argument and returns the information in matrix form. String variables are
    converted to labels.

    :param path: path of the file to be read.
    :return: Read file
    """
    dir = os.path.dirname(__file__)
    x = np.genfromtxt(dir + "/" + path, delimiter=delimiter, dtype=str) #missing_values=["NaN", "NA"])

    le = preprocessing.LabelEncoder()
    # labels = []
    for i in range(0, x.shape[1]):
        #print(x[:, i])
        if not is_number(x[0, i]):
            le.fit(x[:, i])
            x[:, i] = (le.transform(x[:, i])).astype("float32")
            # if clas and (not clas == i):
                # labels += [i]

    x = x.astype("float32")
    """if not labels == []:
        oh = preprocessing.OneHotEncoder(categorical_features=labels, sparse=False)
        x = oh.fit_transform(x)
    """
    return x
    

# if introduceMD==1, reads data in path, separates class atribute, introduces MDT type MD and writes features and class together.
# if introduceMD==0 reads path and returns classes and attributes separated.

# The first functionality is used for the missing data introduction process. After converting some values to "NaN", it writes them to be read by the imputation processes
# The second one is unused right now

def getMDfiles(path, index, MDT, k, clas, introduceMD, seed):

    """

    :param path: path of the file in which the data to be partially erased can be found
    :param index: DB index
    :param MDT: MDT index
    :param k: instance index
    :param clas: variable index where the class can be found
    :param introduceMD: Whether MD is going to be introduced. See description above.
    :param seed: Seed for random initialization
    :return: Writes 5 versions of the dataset with different MD combinations, aiming for a stratified 5-fold
    cross-validation.
    """

    random.seed(seed)

    x = read("Data1/" + path)

    # Assign the class to other variable.
    y = x[:, clas]
    x = np.delete(x, clas, 1)
    i = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(x, y):
        # Separate
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        if introduceMD == 1:
            # Insert MD
            if MDT % 4 == 0:
                x_train = insertMAR(x_train, 1)
                x_test = insertMAR(x_test, 1)
            elif MDT % 4 == 1:
                x_train = insertMCAR(x_train, 1)
                x_test = insertMCAR(x_test, 1)
            elif MDT % 4 == 2:
                x_train = insertMuOV(x_train, 1)
                x_test = insertMuOV(x_test, 1)
            elif MDT % 4 == 3:
                x_train = insertMIV(x_train, 1)
                x_test = insertMIV(x_test, 1)

        # We always write first the training part and hen the testing part.
        x_tot = np.concatenate((x_train, x_test), axis=0)
        y_tot = np.concatenate((y_train, y_test))

        x_tot = np.column_stack((x_tot,y_tot))

        aux = np.array(path.split("."))
 
        npath = "Data/" + str(index) + "-" + str(MDT) + "-" + str(k) + "-" + str(i) + "." + aux[len(aux)-1]
        i += 1
        df = pd.DataFrame(x_tot)

        df.to_csv(npath, header = False, index = False, na_rep = "NaN")

    return x, y

def getMDfilesPercent(path, index, percent, k, clas, introduceMD, seed):

    """

    :param path: path of the file in which the data to be partially erased can be found
    :param index: DB index
    :param MDT: MDT index
    :param k: instance index
    :param clas: variable index where the class can be found
    :param introduceMD: Whether MD is going to be introduced. See description above.
    :param seed: Seed for random initialization
    :return: Writes 5 versions of the dataset with different MD combinations, aiming for a stratified 5-fold
    cross-validation.
    """

    random.seed(seed)

    x = read("Data1/" + path)

    # Assign the class to other variable.
    y = x[:, clas]
    x = np.delete(x, clas, 1)
    i = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(x, y):
        # Separate
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        if introduceMD == 1:
            x_train = insertMCAR(x_train, percent)
            x_test = insertMCAR(x_test, percent)


        # We always write first the training part and hen the testing part.
        x_tot = np.concatenate((x_train, x_test), axis=0)
        y_tot = np.concatenate((y_train, y_test))

        x_tot = np.column_stack((x_tot,y_tot))

        aux = np.array(path.split("."))

        npath = "DataMD/" + str(index) + "-" + str(percent) + "-" + str(k) + "-" + str(i) + "." + aux[len(aux)-1]
        i += 1
        df = pd.DataFrame(x_tot)

        df.to_csv(npath, header = False, index = False, na_rep = "NaN")

    return x, y
