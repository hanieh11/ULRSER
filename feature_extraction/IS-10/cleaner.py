import os
import numpy as np

files = os.listdir('features')

X = np.zeros((len(files), 1582))

for i, file in enumerate(files):
    print('--------', i, '-----------')
    
    data = open(os.path.join('features', file)).readlines()
    data = data[-1].split(',')[1:-1]

    t = np.array(data).astype(np.float64)
    X[i] = t
    
print(X)
np.save('IS10_Emo-DB.npy', X)
    
