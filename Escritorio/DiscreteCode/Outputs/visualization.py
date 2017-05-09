import pandas as pd
import sys
import os
import statistics as st
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from subprocess import check_output
import shutil

MDT = ["MAR", "MCAR", "MuOV", "MIV"]
methods = ['mean', 'median', 'most_frequent', 'lvcf', 'interpolation', 'HD', 'mice', 'EM']
classifiers = ["Regresion L1", "Regresion L2", "LDA", "QDA",  "C-SVM", "RBFN", "Naive Bayes", "GradientBoost.", "RandomForest", "CART", "1-NN", "3-NN"]
dir = ["Low", "Med", "High"]

def read(path):

    x = pd.read_csv(path, header=None)
    x = x.as_matrix()
  
    return x

def filterRowsBy(matrix, crit, val): ##  0 BD - 1 MDT - 2 Instance - 3 IM - 4 Classifier

    return(matrix[matrix[:, crit]==val])
    
def extracTuples(matrix, a1, a2):
    
    axis1 = set(matrix[:,a1])
    axis2 = set(matrix[:,a2])
    
    m = np.empty([len(axis1), len(axis2)])
    
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            aux = filterRowsBy(matrix, a1, i)
            aux = filterRowsBy(aux, a2, j)
            m[i,j] = aux.shape[0]
    
    return(m)
    
def plot(m):

    #
    # Create a figure for plotting the data as a 3D histogram.
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.
    #
    x_data, y_data = np.meshgrid( np.arange(m.shape[1]),
                                  np.arange(m.shape[0]) )
    #
    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    #
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = m.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              1, 1, z_data )
    #
    # Finally, display the plot.
    #
    plt.show()
    
def plot2(data, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(0, data.shape[0]):
        xs = np.arange(data.shape[1])
        ys = data[i,:]

        ax.bar(xs, ys, zs=i, zdir='y', color=colors[i], alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(path)
    

    
if __name__ == '__main__':
    for i in range(0,len(dir)):
        if os.path.exists(dir[i]):
            shutil.rmtree(dir[i])
        os.makedirs(dir[i])
            
    
    if len(sys.argv) == 1:
        paths = ["Output" + str(x) + ".csv" for x in range(0,10)]
    else:
        paths = [""] * (len(sys.argv)-1)
        for i in range(1, len(sys.argv)):
            paths[i-1] = "Output" + str(sys.argv[i]) + ".csv"
    #print(paths)
    
    lsum = np.empty([0,6])
    msum = np.empty([0,6])
    hsum = np.empty([0,6])
    
    for path in paths:
        
        x = read(path)
        
        """for c in range(0, len(classifiers)):
        
            x1 = filterRowsBy(x, 4, c)
            
            print(x1)"""
        x = x[x[:,5].argsort()]
        
        l = x[0:x.shape[0]/3,:]
        m = x[x.shape[0]/3:x.shape[0]/3*2,:]
        h = x[x.shape[0]/3*2:x.shape[0],:]

        lsum = np.concatenate((lsum,l))
        msum = np.concatenate((msum,m))
        hsum = np.concatenate((hsum,h))
        
    for i in range(0, len(MDT)):
        ml = filterRowsBy(lsum, 1, i)
        mm = filterRowsBy(msum, 1, i)
        mh = filterRowsBy(hsum, 1, i)
        
        ml = np.array(extracTuples(ml, 3, 4))
        mm = np.array(extracTuples(mm, 3, 4))
        mh = np.array(extracTuples(mh, 3, 4))
        
        plot2(ml, dir[0] + "/" + MDT[i] + ".png")
        plot2(mm, dir[1] + "/" + MDT[i] + ".png")
        plot2(mh, dir[2] + "/" + MDT[i] + ".png")
        
        aux = check_output('Rscript.exe visualization.r ' + dir[0] + "/" + MDT[i] + ".pdf" + " "+ str(ml.shape[0]) + " " + str(ml.shape[1]) + " " + " ".join(map(str, np.ravel(ml))), shell=True)
        aux = check_output('Rscript.exe visualization.r ' + dir[1] + "/" + MDT[i] + ".pdf" + " "+ str(mm.shape[0]) + " " + str(mm.shape[1]) + " " + " ".join(map(str, np.ravel(mm))), shell=True)
        aux = check_output('Rscript.exe visualization.r ' + dir[2] + "/" + MDT[i] + ".pdf" + " "+ str(mh.shape[0]) + " " + str(mh.shape[1]) + " " + " ".join(map(str, np.ravel(mh))), shell=True)
        #print(aux)