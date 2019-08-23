from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

csv_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Tensorflow/Test0822/")
ltk_list = pd.read_csv(csv_dir+"/test0822.csv",",")
pd_list = pd.DataFrame(ltk_list)

Past_Data = np.array(pd_list[0:np.nonzero(pd_list.isnull()['kp_0h'])[0][0]])
Future_Data = np.array(pd_list[np.nonzero(pd_list.isnull()['kp_0h'])[0][-1]+1:])
Null_Data = np.array(pd_list.loc[pd_list.isnull()['kp_0h'],:])

def split_Data(Data_list, split_Number):
    temp_list = []
    for i in range(len(Data_list) - split_Number + 1):
        subset = Data_list[i:(i+split_Number)]
        temp_list.append([item for item in subset])
    temp_list = np.array(temp_list)
    temp_list = temp_list[:,:,1:]
    return temp_list

Past_5th_Data = split_Data(Past_Data, Null_Data.shape[0])
Future_5th_Data = split_Data(Future_Data, Null_Data.shape[0])
# Create_5th_Data = np.concatenate((Past_5th_Data, Future_5th_Data), axis=0)
Past_5th_Data_Predict = Past_5th_Data[-1]
Future_5th_Data_Predict = Future_5th_Data[0]

def splite_data(_5th_Data, name = "Past"):
    if name == "Past":
        X_Data = _5th_Data[:-5]
        Y_Data = _5th_Data[5:]
    elif name == "Future":
        X_Data = _5th_Data[5:]
        Y_Data = _5th_Data[:-5]
    x_train, x_test, y_train, y_test = train_test_split(
        X_Data, Y_Data, random_state=66, test_size = 0.3, shuffle = True
    )
    return x_train, x_test, y_train, y_test

x_past_train, x_past_test, y_past_train, y_past_test = splite_data(Past_5th_Data, "Past")
x_future_train, x_future_test, y_future_train, y_future_test = splite_data(Future_5th_Data, "Future")


def reshape_data(y_train, y_test):
    y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1] * y_train.shape[2]))
    y_test = np.reshape(y_test, (y_test.shape[0],y_test.shape[1] * y_test.shape[2]))
    return y_train, y_test

y_past_train, y_past_test = reshape_data(y_past_train, y_past_test)
y_future_train, y_future_test = reshape_data(y_future_train, y_future_test)

# print(x_past_train.shape)
# print(y_past_train.shape)
# print(x_past_test.shape)
# print(y_past_test.shape)
# print(x_future_train.shape)
# print(y_future_train.shape)
# print(x_future_test.shape)
# print(y_future_test.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

def Build_model():
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(5,8)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.compile(optimizer='adam', loss='mse',metrics=['acc'])
    return model

def Fit_model(x_train, y_train, x_test, y_test, Data_Predict):
    model = Build_model()
    model.fit(x_train, y_train, batch_size=200, epochs=3000, verbose=1)
    Data_Predict = np.reshape(Data_Predict, (1,5,8))
    _, acc = model.evaluate(x_test, y_test)
    y_predict = model.predict(Data_Predict)
    y_predict = np.reshape(y_predict, (5,8))
    return acc, y_predict
past_acc, Past_Pre_Data = Fit_model(x_past_train, y_past_train, x_past_test, y_past_test, Past_5th_Data_Predict)
# future_acc, Futrue_Pre_Data = Fit_model(x_future_train, y_future_train, x_future_test, y_future_test, Future_5th_Data_Predict)

print("==========================================================")
print("ACC:",past_acc,"\nPredict:\n", Past_Pre_Data)
print("==========================================================")
# print("ACC:",future_acc,"\nPredict:\n", Futrue_Pre_Data)

# # 과거 미래의 평균 리턴 [test_0822_ltk.csv, test_0822_ltk4.csv]
# SUM_5th_Data_Predict = (Past_Pre_Data + Futrue_Pre_Data) / 2
# print("==========================================================")
# print(SUM_5th_Data_Predict)
# print("==========================================================")
# predict_result = np.around(SUM_5th_Data_Predict.astype(np.double), 0)
# predict_result = predict_result.astype(int)
# print(np.around(SUM_5th_Data_Predict.astype(np.double), 0))
# print(predict_result)

# 과거 예측 데이터 [test_0822_ltk3.csv, test_0822_ltk5.csv]
predict_result = np.round(Past_Pre_Data.astype(np.double),0)
predict_result = predict_result.astype(int)

# 미래 예측 데이터 [test_0822_ltk2.csv]
# predict_result = np.round(Futrue_Pre_Data.astype(np.double),0)
# predict_result = predict_result.astype(int)

print(predict_result)
np.savetxt(csv_dir+"/test0822_ltk6.csv", predict_result, delimiter=",")