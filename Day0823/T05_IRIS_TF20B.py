from sklearn.model_selection import train_test_split
import numpy as np
import os

Node_number = 32
def Build_Model(X, train_num):
    Model_Layer = tf.layers.dense(X, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, Node_number, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, train_num, activation=tf.nn.relu)
    return Model_Layer;

save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")
cancer_data = np.load(save_dir+"/iris2_data.npy")
x_data = cancer_data[:,0:-1]
y_data = cancer_data[:,[-1]]
nb_classes = 3

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=77, test_size = 0.25
)

import tensorflow as tf
tf.set_random_seed(777)

train_num = 4
test_num = 1
X = tf.placeholder(tf.float32, shape=[None, train_num])
Y = tf.placeholder(tf.int32, shape=[None, test_num])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot:",  Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:",  Y_one_hot)

logits = Build_Model(X, nb_classes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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