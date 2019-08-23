import tensorflow as tf
tf.set_random_seed(777)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

W = tf.Variable([0.3], name="weight")
b = tf.Variable([-0.3], name="bias")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W + b

# cost / loss function
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# optimizer
# 경사 하강법 GradientDescent
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 훈련
    for step in range(1000):
        sess.run(train, {X: x_train, Y: y_train})
    W_val, b_val, cost_val = sess.run([W,b, cost], feed_dict={X:x_train, Y: y_train})
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")
    