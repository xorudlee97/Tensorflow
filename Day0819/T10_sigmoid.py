import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# [행무시, 열]
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# 시그모이드 함수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost 앞에 -가 잇다.
cost  = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                        tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            print(step, cost_val)
    hy_Val, predicted_Val, acc = sess.run(
        [hypothesis, predicted, accuracy],
        feed_dict = {X: x_data, Y: y_data}
    )
    print(
        "\nHypothesis:\n", hy_Val,
        "\nCorrect (Y):", predicted_Val,
        "\nAccuracy:", acc
    )

