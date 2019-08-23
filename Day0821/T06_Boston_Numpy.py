
from sklearn.datasets import load_boston
import numpy as np
import os

file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")

boston = load_boston()

x = boston.data
y = boston.target
x = np.array(x)
y = np.array(y)
y = np.reshape(y, (y.shape[0], 1))

np.save(save_dir+"/boston_train.npy",x)
np.save(save_dir+"/boston_test.npy",y)

boston_train = np.load(save_dir+"/boston_train.npy")
boston_test = np.load(save_dir+"/boston_test.npy")
