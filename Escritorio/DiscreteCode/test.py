###################################################

##This file contains necessary functions to perform an statistical test to
##our data. An output file (representing DB), a classifier (0-11) and a 
##MDT (0-3) will be passed as parameters.
##Example: "python test.py Output2.csv 0 0"

###################################################
import argparse
import pandas as pd
import inoutmd as io
import numpy as np
from tabulate import tabulate
from scipy.stats.mstats import kruskalwallis

mdtypes = 4
classifiers = 12

##Given a classifier id and a matrix, returns the matrix's rows containing accuracies
##calculated using c-th classifier 
def filterRowsByClassifier(c, matrix):

    return(matrix[matrix[:, 4]%classifiers==c])
    
##Given a MDT and a matrix, returns the matrix's rows containing accuracies
##calculated from DBs affected with MDT-th type  
def filterRowsByMD(mdt, matrix):

    return(matrix[matrix[:, 1]%mdtypes==mdt])
    
##Given a matrix and two columns returns a new matrix using the columns'
##information, containing their accuracies    
def matrixTransformation(matrix, x,y):
    
    ##Amount of rows in our new matrix
    xAxes = np.unique(matrix[:,x])
    ##Amount of columns in our new matrix
    yAxes = np.unique(matrix[:,y])
    ##Matrix construction
    nMatrix = np.empty([len(xAxes), len(yAxes)])
    for i in range(0,len(matrix)):  ##For each row
        xAux = matrix[i,x]          ##Get x and y coordinates in our new matrix
        yAux = matrix[i,y]          ##which correspond with column values
        nMatrix[xAux, yAux] = matrix[i,5] ##5 represents the accuracy column
    #print(tabulate(nMatrix))
    return(nMatrix)
    
    ##Given a matrix and certain columns, returns H-statistic and p-value.
    ##If no columns are provided, all of them will be analyzed.
def test(matrix, columns = -1):

    if columns == -1:
        columns = range(0, matrix.shape[1])
    ##Make the matrix understandable for the test function
    aux = []
    for i in columns:
        aux.append(matrix[:,i])
        
    return(kruskalwallis(aux))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('char', metavar='char', type=open, nargs=1, help='Input file')
        
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(50),
         nargs=2, help='Select classifier and MDtype')
    
    args = parser.parse_args()
    path = args.char[0]  ##path for the output file to be treated
    c = args.integers[0]    ##Classifier
    MDT = args.integers[1]  ##Missing data type
    
    matrix = np.loadtxt(path, delimiter=",") ##Read file
    
    matrix = filterRowsByClassifier(c, matrix) ##Filter by selected classifier
    matrix = filterRowsByMD(MDT, matrix) ##Filter by selected MDT
    
    matrix = matrixTransformation(matrix, 2, 3) ##Transform matrix selecting third column (instances) as rows
                                                ##and fourth (IM) as columns.                                           
    
    columnPairs = [] ##Will contain samples originated from the same distribution
    print(test(matrix)[1])
    if(test(matrix)[1]<0.05/matrix.shape[0]): ##If the first test gives positive (bonferroni correction applied)
        for i in range(0, matrix.shape[1]):         ##For each column
            for j in range(i+1, matrix.shape[1]):   ##Try with every partner not tested yet.
                if(test(matrix, (i,j))[1]<0.05/matrix.shape[0]): ##Compare each pair of columns (bonferroni again applied)
                    columnPairs.append([i,j])
    print(columnPairs)
    
        