import tensorflow as tf
tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
# 경사 하강법 GradientDescent
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 훈련
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            # 텐서에서는 모델명과 그래프를 명시
            [train, cost, W, b], 
            feed_dict={X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]}
        )
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    
    # 모델 출력
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))