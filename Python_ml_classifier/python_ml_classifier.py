from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

import streamlit as st


def generateXYdata(dataType="OneInformative_OneCluster"):
    if dataType == "OneInformative_OneCluster":
        X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    elif dataType == "TwoInformative_OneCluster":
        X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    elif dataType == "TwoInformative_TwoCluster":
        X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2)
    elif dataType == "TwoInformative_OneCluster_MultiClass":
        X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)
    elif dataType == "ThreeBolbs":
        X1, Y1 = make_blobs(n_features=2, centers=3)
    elif dataType == "ThreeGaussian":
        X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
    elif dataType == "Iris":
        iris = datasets.load_iris()
        X1 = iris.data[:, [0, 2]]
        Y1 = iris.target
    return X1,Y1

def setClassifier(classifierType="DecisionTree"):
    if classifierType == "DecisionTree":
        clf = DecisionTreeClassifier(max_depth=4)
    elif classifierType == "KNeighbors":
        clf = KNeighborsClassifier(n_neighbors=7)
    elif classifierType == "SVC":
        clf = SVC(gamma=0.1, kernel="rbf", probability=True)
    elif classifierType == "Voting(Dtree_KN_SVC)":
        clf1 = DecisionTreeClassifier(max_depth=4)
        clf2 = KNeighborsClassifier(n_neighbors=7)
        clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)
        clf = VotingClassifier(
        estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],
        voting="soft",weights=[2, 1, 2],)
    elif classifierType == "SVC_liner":
        clf = SVC(kernel="linear", C=0.025)
    elif classifierType == "GaussianProcess":
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif classifierType == "RandomForest":
        clf = RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1)
    elif classifierType == "NeuralNet":
        clf = MLPClassifier(alpha=1, max_iter=1000)
    elif classifierType == "AdaBoost":
        clf = AdaBoostClassifier()
    elif classifierType == "NaiveBayes":
        clf = GaussianNB()
    elif classifierType == "QDA":
        clf = QuadraticDiscriminantAnalysis()
    return clf


dataSetName = st.selectbox(
    'dataSet',
    ("OneInformative_OneCluster", "TwoInformative_OneCluster", 
    "TwoInformative_TwoCluster", "TwoInformative_OneCluster_MultiClass",
    "ThreeBolbs", "ThreeGaussian",
    "Iris"))

classifierNames = st.multiselect(
        "Choose Classifier", 
        ["KNeighbors","SVC","NaiveBayes", "RandomForest", "NeuralNet", "DecisionTree",
        "Voting(Dtree_KN_SVC)","SVC_liner", "GaussianProcess", "AdaBoost", "QDA"],
        ["KNeighbors","SVC","NaiveBayes", "RandomForest"]
    )

X,y = generateXYdata(dataSetName)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

fig, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))

classifierNames4 = classifierNames
numElm = 4 - len(classifierNames)
if numElm > 0: 
    for i in range(numElm): classifierNames4.append("")

for idx, clfName in zip(
    product([0, 1], [0, 1]),
    classifierNames4,
):

    if len(clfName) == 0: continue

    clf = setClassifier(clfName)
    clf.fit(X, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    axarr[idx[0], idx[1]].set_title(clfName)

# #plt.show()
st.pyplot(fig)

st.button("Re-run")