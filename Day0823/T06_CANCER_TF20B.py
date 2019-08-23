from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import tensorflow as tf
tf.set_random_seed(666)

Node_number = 32
def Build_Model(X, test_num):
    Model_Layer = tf.layers.dense(X, Node_number, activation=tf.sigmoid)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, test_num, activation=tf.sigmoid)
    return Model_Layer;

save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")
cancer_data = np.load(save_dir+"/cancer_data.npy")
x_data = cancer_data[:,0:-1]
y_data = cancer_data[:,[-1]]



print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=77, test_size = 0.3
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

train_num = 30
test_num = 1
X = tf.placeholder(tf.float32, shape=[None, train_num])
Y = tf.placeholder(tf.float32, shape=[None, test_num])

logits = Build_Model(X, test_num)

cost = -tf.reduce_mean(Y * tf.log(logits) + (1 - Y) * tf.log(1 - logits))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy  = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        _, cost_val, acc_Val = sess.run(
            [optimizer, cost, accuracy], 
            feed_dict={X:x_train, Y: y_train}
        )
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tACC: {:.2%}".format(step, cost_val, acc_Val))
    
    pred = sess.run(prediction, feed_dict={X:x_test})
    for p , y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))
    
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess,
            feed_dict={X:x_test, Y:y_test}
        )
    )
