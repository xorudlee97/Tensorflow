from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import tensorflow as tf

save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")
x_data = np.load(save_dir+"/boston_train.npy")
y_data = np.load(save_dir+"/boston_test.npy")

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=77, test_size = 0.5
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

train_num = 13
test_num = 1
X = tf.placeholder(tf.float32, shape=[None, train_num])
Y = tf.placeholder(tf.float32, shape=[None, test_num])

def create_Relu_Layer(input_node, output_Node, layer_hypothesis, weight_name="weihgt", bias_name="bias"):
    W = tf.get_variable(weight_name, shape=[input_node,output_Node], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_Node]), bias_name)
    logits = tf.matmul(layer_hypothesis, W) + b
    hypothesis = tf.nn.relu(logits)
    return hypothesis, logits, W, b
Node_number = 16
hypothesis, logits, W, b = create_Relu_Layer(train_num,Node_number,X, "relu1")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu2")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu3")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu4")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,train_num,hypothesis, "relu5")
hypothesis, logits, W, b = create_Relu_Layer(train_num,test_num,hypothesis, "relu6")

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_Val, _ = sess.run(
            [cost, train],
            feed_dict={X:x_train, Y: y_train}
        )
        if step % 100 == 0:
            print(step, "Cost: ", cost_Val)
    prediction = sess.run(hypothesis, feed_dict={X:x_test})
    r2_y_predict = r2_score(y_test, prediction)
    print("R2 :", r2_y_predict, "\nRMSE : ", RMSE(y_test, prediction))
