from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

n_neighbors = 20

def KNN_clf(X_train, X_test, y_train, y_test):
    global n_neighbors
    cls = KNN(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    return cls
import numpy as np
import os
from sklearn.preprocessing import StandardScaler as ZNormalization
from ULRSER.dimentionality_reduction import *
from ULRSER.classification import *
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
import json
import sklearn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler as ZNormalization
from ULRSER.dimentionality_reduction import *
from ULRSER.classification import *
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
import json
import sklearn


def unison_shuffle(mats):
    assert len(mats) > 2
    p = np.random.permutation(mats[0].shape[0])
    return (mat[p] for mat in mats)


def load_labels():
    emo_code = {'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6}
    files = os.listdir('../Emo-DB/wav')
    files.sort()
    y = np.zeros(len(files))
    for i, file in enumerate(files):
        y[i] = emo_code[file[5]]
    return y


def KNN_clf(X_train, y_train):
    n_neighbors = 17
    cls = KNN(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    return cls

IS10_X = np.load('IS10_Emo-DB.npy')
RQA_X = np.load('RQA_Emo-DB.npy')
fused_X = np.hstack((IS10_X, RQA_X))
y = load_labels()

IS10_X, RQA_X, fused_X, y = unison_shuffle((IS10_X, RQA_X, fused_X, y))

scalar = ZNormalization()
fused_X = scalar.fit_transform(fused_X)

from ULRSER.dimentionality_reduction import LLE_dr

# fused_X = PCA_dr(fused_X, [50])[0]

X_train = fused_X[:450]
y_train = y[:450]
X_test = fused_X[450:]
y_test = y[450:]

cls = KNN_clf(X_train, y_train)
pred = cls.predict(X_test)
for i in range(X_test.shape[0]):
    print('y:', y[i], 'pred:', pred[i])
