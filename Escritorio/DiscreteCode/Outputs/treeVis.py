import pandas as pd
import sys
import os
import numpy as np
from scipy import stats
from IPython.display import Image
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

MDT = ["MAR", "MCAR", "MuOV", "MIV"]
methods = ['mean', 'median', 'most_frequent', 'lvcf', 'interpolation', 'HD', 'mice', 'EM']
classifiers = ["Regresion L1", "Regresion L2", "LDA", "QDA",  "C-SVM", "RBFN", "Naive Bayes", "GradientBoost.", "RandomForest", "CART", "1-NN", "3-NN"]

def read(path):

    x = pd.read_csv(path, header=None)
    x = x.as_matrix()
  
    return x

def filterRowsBy(matrix, crit, val): ##  0 BD - 1 MDT - 2 Instance - 3 IM - 4 Classifier

    return(matrix[matrix[:, crit]==val])
    
def clas(paths):

    xtot = np.empty([0,6])

    for path in paths:
        
        x = read(path)
        xtot = np.concatenate((xtot,x))
        
    xtot = xtot[xtot[:,5].argsort()]
    
    cl = np.zeros((xtot.shape[0],1))
    
    for i in range(0, xtot.shape[0]):
        #print(xtot.shape[1]/3)
        cl[i] = i//(xtot.shape[0]/3)
        
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xtot[:,[0,1,3,4]], cl)
        
     
    dot_data = StringIO()  
    tree.export_graphviz(clf, feature_names = ["BD", "MDT", "IM", "Classifier"], class_names = ["Low", "Med", "High"])  

        
    print (xtot)
    
if __name__ == '__main__':            
    
    if len(sys.argv) == 1:
        paths = ["Output" + str(x) + ".csv" for x in range(0,10)]
    else:
        paths = [""] * (len(sys.argv)-1)
        for i in range(1, len(sys.argv)):
            paths[i-1] = "Output" + str(sys.argv[i]) + ".csv"

    clas(paths)

    

        
        