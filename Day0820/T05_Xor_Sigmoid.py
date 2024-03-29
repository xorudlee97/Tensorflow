import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost  = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_Val, w_Val = sess.run(
            [train, cost, W], feed_dict={X:x_data, Y:y_data}
        )
        if step % 100 == 0:
            print(step, cost_Val, w_Val)
    
    hy_Val, correct, acc = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data}
    )
    print("\nHypothesis:", hy_Val, "\nCorrect:",correct, "\nAccuracy:",acc)