

# tf.constant_initializer()
# tf.zeros_initializer()
# tf.random_uniform_initializer()
# tf.random_normal_initializer()
# tf.contrib.layers.xavier_initializer()


# W = tf.get_variable("W", shape=[?,?],
#                     initializer=tf.random_uniform_initializer())
# b = tf.Variable(tf.random_normal([512]))
# L = tf.nn.relu(tf.matmul(X, W) + b)
# L = tf.nn.dropout(L, keep_prob=keep_prob)

import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

# print(mnist.train.images)
# print(mnist.test.labels)
print(mnist.train.images.shape)
print(mnist.test.labels.shape)
# print(type(mnist.train.images))

#################################################
# 코딩하시오. X, Y, W, b hypothesis, cost, train
#################################################
train_num = 28 * 28
test_num = 10
X = tf.placeholder(tf.float32, shape=[None, train_num])
Y = tf.placeholder(tf.float32, shape=[None, test_num])

def create_Relu_Layer(input_node, output_Node, layer_hypothesis, weight_name="weihgt", bias_name="bias"):
    W = tf.get_variable(weight_name, shape=[input_node,output_Node], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_Node]), bias_name)
    logits = tf.matmul(layer_hypothesis, W) + b
    hypothesis = tf.nn.relu(logits)
    return hypothesis, logits, W, b

def create_Sigmoid_Layer(input_node, output_Node, layer_hypothesis, weight_name="weihgt", bias_name="bias"):
    W = tf.Variable(tf.random_normal([input_node, output_Node]), weight_name)
    b = tf.Variable(tf.random_normal([output_Node]), bias_name)
    logits = tf.matmul(layer_hypothesis, W) + b
    hypothesis = tf.sigmoid(logits)
    return hypothesis, logits, W, b

Node_number = 30
hypothesis, logits, W, b = create_Sigmoid_Layer(train_num,Node_number, X)
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu1")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu2")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu3")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu4")
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,10,hypothesis)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, 
                                         labels=tf.stop_gradient([Y])))
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test Model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 20
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        
        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_Val = sess.run([train, cost], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += cost_Val / num_iterations

        print("Epoch: {:04d}, cost: {:.9f}".format(epoch +1, avg_cost))
    print("Learing finished")

    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess,
            feed_dict={X:mnist.test.images, Y:mnist.test.labels}
        )
    )
    
    r = random.randint(0, mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(
            tf.argmax(hypothesis, 1),
            feed_dict={X:mnist.test.images[r: r + 1]}
        )
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28,28),
        cmap="Greys",
        interpolation="nearest"
    )
    plt.show()


