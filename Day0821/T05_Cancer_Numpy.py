
from sklearn.datasets import load_breast_cancer
import numpy as np
import os

file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target
x = np.array(x)
y = np.array(y)

np.save(save_dir+"/cancer_train.npy",x)
np.save(save_dir+"/cancer_test.npy",y)

cancer_train = np.load(save_dir+"/cancer_train.npy")
cancer_test = np.load(save_dir+"/cancer_test.npy")
