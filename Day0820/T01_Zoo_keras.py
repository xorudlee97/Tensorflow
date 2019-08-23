from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os

file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
xy_data = np.loadtxt(file_dir+"/data-04-zoo.csv", delimiter=",", dtype=np.float32)

x_data = xy_data[:,:-1]
y_data = xy_data[:,[-1]]
input_list_shape= 16
nb_classes = 7

y_data = to_categorical(y_data)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=66, test_size = 0.2
)

from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(nb_classes, input_shape=(input_list_shape,), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=5,epochs=3000)

loss, acc = model.evaluate(x_test, y_test)
print(acc)