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

percent = 7  # This work considers introducing 7% of missing values in MD free DBs

# ####################### Missing Data inserting functions ###################


# This function introduces MCAR type MD.
def insertMCAR(data):
    x = data.shape[0]
    y = data.shape[1]
    for i in range(0, x*y*percent//100):
        data[random.randint(0,x-1), random.randint(0,y-1)] = "NaN"  # It randomly selects two indices from the table and assign a "NaN"
         
    return data


# This function introduces MuOV type MD
def insertMuOV(data):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), 3)  # Select the features losing their values
    while len(obs) < (percent*x*y/100)//3:
        obs.append(random.randint(0, x-1))  # Since the "causative" is unobserved, the observations missing their values are selected randomly
    for i in range(0, len(obs)):  # For the selected observations (above)
        for j in range(0, len(y1)):  # For the selected features
            data[obs[i], y1[j]] = "NaN"  # "NaN" assigning
         
    return data


# This function introduces MIV type MD
def insertMIV(data):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), 4)  # Select which features will lose values
    for i in range(0, len(y1)):
    
        # Auxiliary variable, to select the observations losing values, without modifying them yet.
        auxy = copy.copy(data[:, y1[i]])
        while len(obs) < (percent*y*x/100)//4:
            obs.append(np.argmin(auxy))
            auxy[obs[len(obs)-1]] = 999999
        
        for j in range(0, len(obs)):
            data[obs[j], y1[i]] = "NaN"  # "NaN" assigning for these positions
        obs = []
    return data


# This function introduces MAR type MD
def insertMAR(data):
    x = data.shape[0]
    y = data.shape[1]
    obs = []
    y1 = random.sample(range(0, y), 4)  # First element in y1 will be the "causative" variable, remaining three will lose values
    
    # Auxiliary causative variable, to select the observations losing values, without modifying the causative.
    auxy = copy.copy(data[:, y1[0]])
    while len(obs) < (percent*x*y/100)//3:
        obs.append(np.argmin(auxy))
        auxy[obs[len(obs)-1]] = 999999
    
    for i in range(0, len(obs)):
        for j in range(1, len(y1)):
            data[obs[i], y1[j]] = "NaN"  # "NaN" assigning
         
    return data


# ####################### In/Out functions ######################################
    

# if introduceMD==1, reads data in path, separates class atribute, introduces MDT type MD and writes features and class together.
# if introduceMD==0 reads path and returns classes and attributes separated.

# The firts functionality is used for the missing data introduction process. After converting some values to "NaN", it writes them to be read by the imputation processes
# The second one is unused right now

def getMDfiles(path, index, MDT, k, clas, introduceMD):
    dir = os.path.dirname(__file__)

    x = pd.read_csv(dir + "/Data1/" + path, header=None)
    x = x.as_matrix()
    
    # Assign the class to other variable.
    y = x[:, clas]
    x = np.delete(x, clas, 1)

    x = x.astype(float)
    if introduceMD == 1:
        # Insert MD
        if MDT % 4 == 0:
            x = insertMAR(x)
        elif MDT % 4 == 1:
            x = insertMCAR(x)
        elif MDT % 4 == 2:
            x = insertMuOV(x)
        elif MDT % 4 == 3:
            x = insertMIV(x)
        
        # Write file for R imputation methods

        x = np.column_stack((x,y))

        aux = np.array(path.split("."))
 
        npath = "Data/" + str(index) + "-" + str(MDT) + "-" + str(k) + "." + aux[len(aux)-1]
    
        df = pd.DataFrame(x)

        df.to_csv(npath, header = False, index = False, na_rep = "NaN")

    return x, y
        

# Simply read. Used to read imputed files.
def read(path):
    dir = os.path.dirname(__file__)
    x = pd.read_csv(dir + "/" + path, header=None, delimiter=",")
    x = x.as_matrix()
  
    return x
