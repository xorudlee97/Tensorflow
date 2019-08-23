import numpy as np

# 1. 훈련 데이터 y = wx + b
# 그래프 그리기
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 여기서 Layer의 깊이와 노드의 갯수를 조절한다. 
# [취미생활:데이터 모델링]
# 1차원 직선함수 구하기
model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
from keras.optimizers import Adam
optimizer = Adam(lr=0.009)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# epochs 위의 모델을 100번 돌린다.
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print("mse: ", mse)
pred1  = model.predict([1.5,2.5,3.5])
print(pred1)

'''
===========================================================================
SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

Arguments

lr: float >= 0. Learning rate.
momentum: float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
decay: float >= 0. Learning rate decay over each update.
nesterov: boolean. Whether to apply Nesterov momentum.
===========================================================================
RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

Arguments

lr: float >= 0. Learning rate.
rho: float >= 0.
epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
decay: float >= 0. Learning rate decay over each update.
===========================================================================
Adagrad(lr=0.01, epsilon=None, decay=0.0)

Arguments

lr: float >= 0. Initial learning rate.
epsilon: float >= 0. If None, defaults to K.epsilon().
decay: float >= 0. Learning rate decay over each update.
===========================================================================
Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

Arguments

lr: float >= 0. Initial learning rate, defaults to 1. It is recommended to leave it at the default value.
rho: float >= 0. Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
decay: float >= 0. Initial learning rate decay.
===========================================================================
Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

Arguments

lr: float >= 0. Learning rate.
beta_1: float, 0 < beta < 1. Generally close to 1.
beta_2: float, 0 < beta < 1. Generally close to 1.
epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
decay: float >= 0. Learning rate decay over each update.
amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".
===========================================================================
Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

Arguments

lr: float >= 0. Learning rate.
beta_1: floats, 0 < beta < 1. Generally close to 1.
beta_2: floats, 0 < beta < 1. Generally close to 1.
epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
decay: float >= 0. Learning rate decay over each update.
===========================================================================
Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

Arguments

lr: float >= 0. Learning rate.
beta_1: floats, 0 < beta < 1. Generally close to 1.
beta_2: floats, 0 < beta < 1. Generally close to 1.
epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
schedule_decay: floats, 0 < schedule_decay < 1.
===========================================================================
'''