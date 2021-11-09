import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from numpy import trapz
import random

def getBinaryClassifiers():
    return [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

def CalClassifiers(classifiers):
    newClas = []
    for i in classifiers:
        newClas.append(CalibratedClassifierCV(i))
    return newClas

def getBinaryDataset():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples = 1000)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return(X, y)

def drawBinaryDataset(dataset):
    x_red = []
    y_red = []
    x_blue =[]
    y_blue =[]
    
    for i in range(dataset[0].shape[0]):
        if dataset[1][i] == 0:
            x_red.append(dataset[0][i][0])
            y_red.append(dataset[0][i][1])
        if dataset[1][i] == 1:
            x_blue.append(dataset[0][i][0])
            y_blue.append(dataset[0][i][1])
    plt.plot(x_red, y_red, 'o', color='red')
    plt.plot(x_blue, y_blue, 'o', color='blue')
    
def splitBinaryDataset(dataset):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    return X_train, X_test, y_train, y_test

def fitBinaryClassifier(classifier, train_x, train_y):
    classifier.fit(train_x, train_y)
    
def fitBinaryClassifiers(classifiers, train_x, train_y):
    for classifier in classifiers:
        fitBinaryClassifier(classifier, train_x, train_y)

def predictBinaryClassifier(classifier, test):
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(test)
    else:
        Z = classifier.predict_proba(test)[:, 1]
    return Z

def predictionTable(classifier, X_test):
    return predictBinaryClassifier(classifier, X_test)

def TPRate(predict, ground_truth, threshold):
    tp = np.sum((predict >= threshold) & (ground_truth == 1))
    true_count = np.sum(ground_truth == 1)
    return tp / true_count


def FPRate(predict, ground_truth, threshold):
    fp = np.sum((predict >= threshold) & (ground_truth == 0)) #remove add.reduce
    false_count = np.sum(ground_truth == 0)
    return fp / false_count

def TP(predict, ground_truth, threshold):
    tp = np.sum((predict >= threshold) & (ground_truth == 1))
    return tp

def FP(predict, ground_truth, threshold):
    fp = np.sum((predict >= threshold) & (ground_truth == 0))
    return fp

def TN(predict, ground_truth, threshold):
    tn = np.sum((predict <= threshold) & (ground_truth == 0))
    return tn

def FN(predict, ground_truth, threshold):
    fn = np.sum((predict <= threshold) & (ground_truth == 1))
    return fn

def precision(predict, ground_truth, threshold):
    return TP(predict, ground_truth, threshold)/(TP(predict, ground_truth, threshold)+FP(predict, ground_truth, threshold))

def recall(predict, ground_truth, threshold):
    return TP(predict, ground_truth, threshold)/(TP(predict, ground_truth, threshold)+FN(predict, ground_truth, threshold))

def drawRocTable(pred, test_y):
    trshs = pred.copy()
    trshs.sort()
    TPRS = [1]
    FPRS = [1]
    for trsh in trshs:
        TPRS.append(TPRate(pred, test_y, trsh))
        FPRS.append(FPRate(pred, test_y, trsh))
    TPRS.append(0)
    FPRS.append(0)
    area = trapz(TPRS, FPRS)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.text(0.2, 0.05, 'Area = '+ str(-1*area), fontsize=15, color='black')
    plt.text(0.2, 0.25, 'ROC', fontsize=15, color='black')
    plt.plot(FPRS, TPRS,  color='green')


def drawRoc(classifier, test_x, test_y):
    pred = predictBinaryClassifier(classifier, test_x)
    drawRocTable(pred, test_y)

def drawPR(classifier, test_x, test_y):
    pred = predictBinaryClassifier(classifier, test_x)
    trshs = pred.copy()
    trshs.sort()
    PrS = []
    ReS = []
    for trsh in trshs:
        PrS.append(precision(pred, test_y, trsh))
        ReS.append(recall(pred, test_y, trsh))
    area = trapz(PrS, ReS)
    plt.text(0.2, 0.05, 'Area = '+ str(-1*area), fontsize=15, color='black')
    plt.text(0.2, 0.15, 'Classifier = '+str(classifier), fontsize=15, color='black')
    plt.text(0.2, 0.25, 'ROC', fontsize=15, color='black')
    plt.plot(ReS, PrS,  color='green')
    plt.xlim([0,1])
    plt.ylim([0,1])
    
def drawPRTable(pred, test_y):
    trshs = pred.copy()
    trshs.sort()
    PrS = []
    ReS = []
    for trsh in trshs:
        PrS.append(precision(pred, test_y, trsh))
        ReS.append(recall(pred, test_y, trsh))
    #PrS.append(1)
    #ReS.append(0)
    area = trapz(PrS, ReS)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.text(0.2, 0.05, 'Area = '+ str(-1*area), fontsize=15, color='black')
    plt.text(0.2, 0.25, 'PR', fontsize=15, color='black')
    plt.plot(ReS, PrS,  color='green')

def predictCoinFlipper(y):
    pred = [1-y[i] if random.randint(1, 4) == 4 else y[i] for i in range(len(y))]
    obj = [random.uniform(0, 1) for i in range(len(y))]
    return pred, obj