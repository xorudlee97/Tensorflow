import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
print("W1 :", W1)
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
# [1 | 1,1 | 1]
# [1 , 1] = 한칸씩 움직여라
# 28, 28을 3, 3으로 자를때 나오는 개수
# 28 - 3 + 1 = 26
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
print("L1 :", L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
print("L1 :", L1)

# L2ImgIn shape= (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
print("W2 :", W2)
# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
print("L2 :", L2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")

# 2.0 부터 케라스 임포트 가능
# from tensoflow.keras import layers
#################케라스 문법으로 사용 가능한 layers##########
L3 = tf.layers.conv2d(L2, 64, [3,3], activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(L3, [2,2], [2,2])
L3 = tf.layers.dropout(L3, 0.7)

L4 = tf.contrib.layers.flatten(L3)
L4 = tf.layers.dense(L4, 256, activation=tf.nn.relu)
L4 = tf.layers.dropout(L4, 0.5)

logits = tf.layers.dense(L4, 10, activation=None)
##########################################################

# L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
# print("L2 :", L2)

# W3 = tf.get_variable("W3", shape=[10, 10], initializer=tf.contrib.layers.xavier_initializer())
# b =  tf.Variable(tf.random_normal([10]))
# logits = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Learning started. It takes sometime.")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            cost_Val, _ = sess.run([cost, optimizer], feed_dict= feed_dict)
            avg_cost += cost_Val /total_batch
        
        print("Epoch: %04d" %(epoch +1), "cost = {:.9f}" .format(avg_cost))
    
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y : mnist.test.labels
    }))

    re = random.randint(0, mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[re:re+1], 1)))
    print("Prediction", sess.run(tf.argmax(logits, 1), feed_dict = {X: mnist.test.images[re:re+1]}))
