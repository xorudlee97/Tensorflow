import tensorflow as tf
tf.set_random_seed(777)

x1_data = [73.,93.,86.,99.,76.]
x2_data = [80.,88.,91.,89.,66.]
x3_data = [75.,93.,70.,90.,100.]

y_data =  [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y  = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name="weight1")
w2 = tf.Variable(tf.random_normal([1]), name="weight2")
w3 = tf.Variable(tf.random_normal([1]), name="weight3")
b =  tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x1 *w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_bal, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict = {x1: x1_data, x2: x2_data, x3:x3_data, Y: y_data}
        )

        if step % 10 == 0:
            print(step, "Cost :", cost_val, "\nPreidiction:\n", hy_bal)