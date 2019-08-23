import numpy as np
import pandas as pd
import os
file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")

def name_class(y):
    return_list = []
    for i in range(len(y)):
        if y[i] == "Iris-setosa":
            return_list.append(0)
        elif y[i] == "Iris-versicolor":
            return_list.append(1)
        elif y[i] == "Iris-virginica":
            return_list.append(2)
    return return_list

iris_data = pd.read_csv(file_dir+"/iris2.csv", encoding="utf-8")


x = np.array(iris_data.iloc[:,:-1])
y = name_class(iris_data.iloc[:,-1])
print(y)

y = np.array(y,dtype=np.int32)
iris2_data = np.c_[x,y]
np.save(save_dir+"/iris2_data.npy",iris2_data)

iris2_data = np.load(save_dir+"/iris2_data.npy")