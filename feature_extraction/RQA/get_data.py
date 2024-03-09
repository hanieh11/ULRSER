from sklearn.externals import joblib
import os
import numpy as np

data = joblib.load('/home/mqod/Desktop/data/segment/berlin RQA2.dat')

files = os.listdir('../../../Emo-DB/wav')
files.sort()

X = np.zeros((len(files), 432))
for i, file in enumerate(files):
    X[i] = data[file[:2]][file[2:7]]['x']

np.save('../../RQA_Emo-DB.npy', X)
# print(data)
# print(files)