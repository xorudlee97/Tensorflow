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

def create_Sigmoid_Layer(input_node, output_Node, layer_hypothesis, activation = "sigmoid"):
    W = tf.Variable(tf.random_normal([input_node, output_Node]), name="weight")
    b = tf.Variable(tf.random_normal([output_Node]), name="bias")
    if activation == "sigmoid":
        logits = tf.matmul(layer_hypothesis, W) + b
        hypothesis = tf.sigmoid(logits)
    return hypothesis, logits, W, b
Node_number = 100
hypothesis,logits, W, b = create_Sigmoid_Layer(train_num,Node_number,X)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,Node_number,hypothesis)
hypothesis,logits, W, b = create_Sigmoid_Layer(Node_number,10,hypothesis)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, 
                                         labels=tf.stop_gradient([Y])))
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test Model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        
        for i in range(num_iterations):
            cost_Val, hy_Val, _ = sess.run(
                [cost, hypothesis, train],
                feed_dict={X:x_data, Y: y_data}
            )
            print(step, "Cost: ", cost_Val,"\nPrediction:\n",hy_Val)
        
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


