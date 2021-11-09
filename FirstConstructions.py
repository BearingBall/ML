import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from itertools import product

def getDataset(fitchNumbers=2):
    iris = datasets.load_iris()
    X = iris.data[:, :fitchNumbers]  # we only take the first two features.
    y = iris.target
    return X, y

def getMetric(power):
    return lambda u,v: distance.minkowski(u, v, p=power, w=None)

def getKernel(type):
    if type == 0:
        return lambda r: 3/4*(1-r*r) if abs(r)<=1 else 0
    elif type == 1:
        return lambda r: 15/16*(1-r*r)*(1-r*r) if abs(r)<=1 else 0
    elif type == 2:
        return lambda r: (1-abs(r)) if abs(r)<=1 else 0
    elif type == 3:
        return lambda r: pow(2*np.pi, -1/2)*np.exp(-1/2*r*r)
    elif type == 4:
        return lambda r: 1/2 if abs(r)<=1 else 0

def w(x1, x2, h, gamma, metric, kernel):
    return gamma*kernel(metric(x1,x2)/h)


def calculateH(X, metric):
    #a = np.arange(X.shape[0]).reshape((X.shape[0],1))*np.ones(X.shape[0]).reshape((X.shape[0],1)).T
    #b = np.arange(X.shape[0]).reshape((X.shape[0],1)).T*np.ones(X.shape[0]).reshape((X.shape[0],1))
    #a = a.reshape(-1).astype(int)
    #b = b.reshape(-1).astype(int)

    #return np.sqrt(np.max(X[a]*X[a]-X[b]*X[b]))
    diff = X[:, np.newaxis,:] - X[np.newaxis,:,:]
    dist = np.sqrt(np.sum((diff**2),-1))
    #print(dist)
    return np.max(dist)



def predict(X, y, weight, whoIsPredicted, metric, kernel, H):
    classWeight = np.zeros(np.max(y)-np.min(y)+1)
    for i in range(len(weight)):
        classWeight[y[i]]+=w(whoIsPredicted,X[i], H ,weight[i], metric, kernel)
    return np.argmax(classWeight)
        
    
def weightFit(X, y, h, metric, kernel, iterationNumber=10):
    weight = np.zeros(len(y))
    for j in range(iterationNumber):
        for i in range(len(weight)):
            if predict(X,y,weight, X[i], metric, kernel, h) != y[i]:
                weight[i] += 1
    return weight
        
def accuracy(X, y, weight, H, metric, kernel, x_test, y_test):
    TPN = []
    TPN = [1 if predict(X,y,weight, x_test[i], metric, kernel, H) == y_test[i] else 0 for i in range(len(y_test)) ]
    return np.sum(TPN)/len(y_test)

def splitDataset(X,y, testSize=.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    return X_train, X_test, y_train, y_test

def DrawDataset(X, y):
    plt.rcParams['figure.figsize'] = [15, 15]
    dataframe = pd.DataFrame(X)
    scatter_matrix(dataframe, c=y,marker='o', s=40,alpha=.8)
    plt.show()
    
def DrawDatasetW(X, y, marker):
    #plt.rcParams['figure.figsize'] = [15, 15]
    
    #non_zero_ids = [i for i in np.arange(weight.shape[0]) if weight[i] > 0]
    #reduced_x = X[non_zero_ids]
    #reduced_y = [y[i] if weight[i]>0 else 2 for i in np.arange(weight.shape[0])]
    #reduced_dataframe = pd.DataFrame(reduced_x)

    #zero_ids = [i for i in np.arange(weight.shape[0]) if weight[i] == 0]
    #unreduced_x = X[non_zero_ids]
    #unreduced_y = y[non_zero_ids]
    #unreduced_dataframe = pd.DataFrame(unreduced_x)
    
    #scatter_matrix(pd.DataFrame(X), c=reduced_y, marker='o', s=40,alpha=.5, diagonal='kde')
    #scatter_matrix(unreduced_dataframe, c=unreduced_y, marker='o', s=40,alpha=.8, diagonal='kde')

    def grid(size):
        return product(range(size), range(size))

    num_features = X.shape[1]
    for i, (f1, f2) in enumerate(grid(num_features)):
        plt.subplot(num_features, num_features, 1 + i)
        if f1 != f2:
            x1_sliced = X[:, f1]
            x2_sliced = X[:, f2]
            plt.scatter(x1_sliced, x2_sliced, c=y, marker=marker, edgecolor='k')
            plt.tight_layout()
    
def ReshuffleDataset(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]

def SimpleTest(X,y, metric, kernel, isDraw = False):
    StrongTest(X,y, metric, kernel, X, y, isDraw)

def StrongTest(X,y, metric, kernel, X_test, y_test, isDraw = False):
    plt.figure(figsize=(10, 10))
    weight = np.ones(len(y))
    print("Accuracy = ", accuracy(X,y,weight, metric,kernel, X_test, y_test))
    print("Non zero coefficients = ", np.count_nonzero(weight))
    if X.shape[1] == 2 and isDraw:
        plt.subplot(2, 1, 1)
        Draw(X,y,weight,metric, kernel, X_test, y_test)
    weight = weightFit(X,y,metric,kernel, 100)
    print("Accuracy = ", accuracy(X,y,weight, metric,kernel, X_test, y_test))
    print("Non zero coefficients = ", np.count_nonzero(weight))
    if X.shape[1] == 2 and isDraw:
        plt.subplot(2, 1, 2)
        Draw(X,y,weight,metric, kernel, X_test, y_test)
    

def Draw(X, y, weight, metric, kernel, x_test, y_test):
    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h=0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    H = calculateH(X, metric)
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i][j] = predict(X,y,weight, (xx[i][j],yy[i][j]), metric, kernel, 2*H)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm_bright, alpha=.8)
    
    for i in range(len(y_test)):
        pred = predict(X,y,weight, x_test[i], metric, kernel, 2*H)
    
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    
    
    
    
#debug code

def accuracyTester(X,y,weight, metric, kernel, x_test, y_test):
    TPN = 0
    H = calculateH(X, metric)
    for i in range(len(y_test)):
        pred = predict(X,y,weight, x_test[i], metric, kernel, 2*H)
        if pred == y_test[i]:
            TPN+=1
        print("x = ", x_test[i], ", y = ", y_test[i], ", pred = ", pred)
    return TPN/len(y_test)

def SimpleTestTester(X,y, metric, kernel):
    StrongTestTester(X,y, metric, kernel, X, y)

def StrongTestTester(X,y, metric, kernel, X_test, y_test):
    weight = np.ones(len(y))
    print("Accuracy = ", accuracyTester(X,y,weight, metric,kernel, X_test, y_test))
    print("Non zero coefficients = ", np.count_nonzero(weight))
    weight = weightFit(X,y,metric,kernel)
    print("Accuracy = ", accuracyTester(X,y,weight, metric,kernel, X_test, y_test))
    print("Non zero coefficients = ", np.count_nonzero(weight))



def HowMuchPurple(weight, y_train):
    Purple = [1 if y_train[i] == 1 and weight[i] > 0 else 0 for i in range(len(weight))]
    return (np.sum(Purple))
