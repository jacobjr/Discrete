###################################################

##This file contains necessary functions to perform an statistical test to
##our data. An output file (representing DB), a classifier (0-11) and a 
##MDT (0-3) will be passed as parameters.
##Example: "python test.py Output2.csv 0 0"
##If parameters out of this range are introduced all range of parameters will be considered

###################################################
import argparse
import pandas as pd
import sys

import statistics as st
import numpy as np
from tabulate import tabulate
from scipy.stats.mstats import kruskalwallis
from scipy.stats import mannwhitneyu
from subprocess import check_output

algorithms = ["Regresion L1", "Regresion L2", "LDA", "QDA",  "C-SVM", "RBFN", "Naive Bayes", "GradientBoost.", "RandomForest", "CART", "1-NN", "3-NN"]
MDT = ["MAR", "MCAR", "MuOV", "MIV"]


##Given a classifier id and a matrix, returns the matrix's rows containing accuracies
##calculated using c-th classifier 
def filterRowsByClassifier(c, matrix):

    return(matrix[matrix[:, 4]%len(algorithms)==int(c)])
    
##Given a MDT and a matrix, returns the matrix's rows containing accuracies
##calculated from DBs affected with MDT-th type  
def filterRowsByMD(mdt, matrix):

    return(matrix[matrix[:, 1]%len(MDT)==int(mdt)])
    
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
    
def produceTestResults(path, c, mdt):
    
    ##If classifier, MDT and path not defined or defined incorrectly, consider all of them
    if c not in range(0, len(algorithms)):
        cs = range(0,len(algorithms))
    else:
        cs = [c]
    if mdt not in range(0, len(MDT)):
        MDTs = range(0, len(MDT))
    else:
        MDTs = [mdt]
    if path[0:6] == "Output" and path[7:11] == ".csv" and path[6] in str(range(0,10)):
        paths = [path]
    else:
        paths = ["Output" + str(x) + ".csv" for x in range(0,10)]

    #return(cs,MDTs, paths)
    
    text_file = open("Tests.csv", "a")
    for mdt in MDTs:
        table = np.zeros((8,8))
        for path in paths:
            m = np.loadtxt(path, delimiter=",") ##Read file
            text_file.write("File: " + path)
            
            for c in cs:
            
                matrix = filterRowsByClassifier(c, m) ##Filter by selected classifier

                matrix = filterRowsByMD(mdt, matrix) ##Filter by selected MDT
                
                matrix = matrixTransformation(matrix, 2, 3) ##Transform matrix selecting third column (instances) as rows
                                                            ##and fourth (IM) as columns.
       
                columnPairs = [] ##Will contain samples originated from the same distribution

                ##BD with almost exact accuracies  
                #if (path[6] == "4" or path[6] == "6") and mdt == 1 and c ==5: 
                    #print matrix
                
                if not((path[6] == "4" or path[6] == "6")  and mdt == 1 and c ==5) and (test(matrix)[1]<0.05): ##If the first test gives positive (not considering equal matrices)
                    for i in range(0, matrix.shape[1]):         ##For each column
                        for j in range(i+1, matrix.shape[1]):   ##Try with every partner not tested yet.
                            aux = check_output('Rscript.exe Dunn.r ' + " ".join(map(str, matrix[:,i])) + " 0 " + " ".join(map(str, matrix[:,j])), shell=True)
                            aux = aux[6:]
                            ## Result from R's dunn.test print(float(aux))
                            if(float(aux)<0.05/(matrix.shape[1]*(matrix.shape[1]+1))/2): ##Compare each pair of columns (bonferroni applied)
                                columnPairs.append([i,j, test(matrix, (i,j))[1]])
                                if st.mean(matrix[:,i]) > st.mean(matrix[:,j]):
                                    table[i,j] += 1
                                else:
                                    table[j,i] += 1
                #print("\t" + MDT[mdt] + ", " + algorithms[c])
                    text_file.write("\n\t" + MDT[mdt] + ", " + algorithms[c] + ": " + str(test(matrix)[1]) + "\n\t\t" + "\n\t\t".join("\t".join(map(str,l)) for l in columnPairs) + "\n")
                    
        np.savetxt("tabla" + str(mdt) + ".csv", table, delimiter = ",", fmt='%s')
    text_file.close()
    
    
if __name__ == '__main__':

    """parser = argparse.ArgumentParser()
    parser.add_argument('char', metavar='char', type=open, nargs=1, help='Input file')
        
    parser.add_argument(
        'integers', metavar='int', type=int,
         nargs=2, help='Select classifier and MDtype')
    
    args = parser.parse_args()
    path = args.char[0]  ##path for the output file to be treated
    c = args.integers[0]    ##Classifier
    MDT = args.integers[1]  ##Missing data type"""
    if len(sys.argv) == 1:
        path = ""
        c = -1
        mdt = -1
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        c = -1
        mdt = -1
    elif len(sys.argv) == 3:
        path = sys.argv[1]
        c = sys.argv[2]
        mdt = -1
    else:
        path = sys.argv[1]
        c = sys.argv[2]
        mdt = sys.argv[3]

    produceTestResults(path, int(c), int(mdt))
    
    
    
        