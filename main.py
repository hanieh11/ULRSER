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


IS10_X = np.load('IS10_Emo-DB.npy')
RQA_X = np.load('RQA_Emo-DB.npy')
fused_X = np.hstack((IS10_X, RQA_X))
y = load_labels()

IS10_X, RQA_X, fused_X, y = unison_shuffle((IS10_X, RQA_X, fused_X, y))

feature_X = [IS10_X, RQA_X, fused_X]
features_name = ['IS10', 'RQA', 'fused']

for i in range(len(feature_X)):
    scalar = ZNormalization()
    feature_X[i] = scalar.fit_transform(feature_X[i])

results = {}



bar = IncrementalBar('Computing performances ...', max=len(features_name) * len(DR) * len(target_dimensions_init), suffix='%(percent).1f%% - %(eta)ds')

for X, n in zip(feature_X, features_name):
    plt.figure()
    plt.title(n + ' features')
    results[n] = {}
    for dr in DR:
        results[n][dr.__name__] = {}
        dr_acc = np.zeros(len(target_dimensions_init))
        for i, X_reduced in enumerate(dr(X)):
            scalar = ZNormalization()
            X_reduced = scalar.fit_transform(X_reduced)
            dr_acc[i], results[n][dr.__name__][str(target_dimensions_init[i])] = AVG_acc(X_reduced, y)
            results[n][dr.__name__][str(target_dimensions_init[i])]['Avg'] = dr_acc[i]
            bar.next()

        plt.plot(target_dimensions_init, dr_acc, 'o-', label=dr.__name__)

    original_acc, results[n]['original'] = AVG_acc(X, y)
    plt.hlines(original_acc, 0, 100, colors='k', linestyles='--', label='original')

bar.finish()
with open('result_Knn.json', 'w') as outfile:
    json.dump(results, outfile)
plt.legend()
plt.show()