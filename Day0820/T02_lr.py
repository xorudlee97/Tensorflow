import tensorflow as tf
tf.set_random_seed(777)

x_data = [
    [1,2,3], [1,3,2],
    [1,3,4], [1,5,5],
    [1,7,5], [1,2,5],
    [1,6,6], [1,7,7]
]
y_data = [
    [0,0,1], [0,0,1],
    [0,0,1], [0,1,0],
    [0,1,0], [0,1,0],
    [1,0,0], [1,0,0]
]
x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y* tf.log(hypothesis), axis= 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.224951).minimize(cost)

predict = tf.arg_max(hypothesis, 1)
is_correct  = tf.equal(predict, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_Val, W_val, _ = sess.run(
            [cost, W, optimizer],
            feed_dict= {X:x_data, Y:y_data}
        )
        print(step, cost_Val, W_val)
    
    print("Prediction:", sess.run(predict, feed_dict={X:x_test}))
    print("Acciracy  :", sess.run(accuracy,feed_dict={X:x_test, Y: y_test}))