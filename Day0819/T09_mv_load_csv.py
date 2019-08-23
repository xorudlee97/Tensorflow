import tensorflow as tf
import numpy as np
import os
tf.set_random_seed(777)

file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
xy_data = np.loadtxt(file_dir+"/data-01-test-score.csv", delimiter=",", dtype=np.float32)
x_data = xy_data[:, 0:-1]
y_data = xy_data[:, [-1]]

print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# None, 행무시
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 행렬 곱셈
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis -Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_bal, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict = {X: x_data, Y: y_data}
        )

        if step % 10 == 0:
            print(step, "Cost :", cost_val, "\nPreidiction:\n", hy_bal)