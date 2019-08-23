import numpy as np
import os

save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")
cancer_data = np.load(save_dir+"/cancer_data.npy")
x_data = cancer_data[:,0:-1]
y_data = cancer_data[:,[-1]]

print(x_data.shape, y_data.shape)

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([30, 1]), name="weihgt")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run(
            [cost, train], 
            feed_dict={X:x_data, Y: y_data}
        )
        if step % 200 == 0:
            print(step, cost_val)
    
    hy_Val, pre_Val, acc = sess.run(
        [hypothesis, predicted, accuracy],
        feed_dict={X:x_data, Y:y_data}
    )
    print(
        "\nHypothesis:\n", hy_Val,
        "\nCorrect (Y):", pre_Val,
        "\nAccuracy:", acc
    )
