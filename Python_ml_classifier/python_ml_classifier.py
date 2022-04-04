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
    elif classifierType == "Voting":
        clf = VotingClassifier(
        estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],
        voting="soft",weights=[2, 1, 2],)
    return clf


dataSetName = st.selectbox(
    'dataSet',
    ("OneInformative_OneCluster", "TwoInformative_OneCluster", 
    "TwoInformative_TwoCluster", "TwoInformative_OneCluster_MultiClass",
    "ThreeBolbs", "ThreeGaussian",
    "Iris"))

X,y = generateXYdata(dataSetName)

clf1 = setClassifier("DecisionTree")
clf2 = setClassifier("KNeighbors")
clf3 = setClassifier("SVC")
eclf = setClassifier("Voting")

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

fig, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))

for idx, clf, tt in zip(
    product([0, 1], [0, 1]),
    [clf1, clf2, clf3, eclf],
    ["Decision Tree (depth=4)", "KNN (k=7)", "Kernel SVM", "Soft Voting"],
):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    axarr[idx[0], idx[1]].set_title(tt)

#plt.show()
st.pyplot(fig)

st.button("Re-run")