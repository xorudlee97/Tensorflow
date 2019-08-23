import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weihgt")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X,W)+ b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run(
            [optimizer, cost], 
            feed_dict={X:x_data, Y: y_data}
        )
        if step % 200 == 0:
            print(step, cost_val)
    
    print("====================")
    a = sess.run(hypothesis, feed_dict={X: [[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))

    print("====================")
    b = sess.run(hypothesis, feed_dict={X: [[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))

    print("====================")
    c = sess.run(hypothesis, feed_dict={X: [[1,0,1,1]]})
    print(c, sess.run(tf.argmax(c,1)))

    print("====================")
    all_item = sess.run(hypothesis, feed_dict={X: [[1,11,7,9], [1,3,4,3], [1,3,4,3]]})
    print(all_item, sess.run(tf.argmax(all_item,1)))

'''
텐서 플로우는 그래프로 이루어져 있다.
완성된 텐서는 머신을 통해 머신러닝을 수행한다.
'''

