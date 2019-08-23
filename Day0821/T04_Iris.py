from sklearn.model_selection import train_test_split
import numpy as np
import os

save_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/Numpy/")
cancer_data = np.load(save_dir+"/iris2_data.npy")
x_data = cancer_data[:,0:-1]
y_data = cancer_data[:,[-1]]
nb_classes = 3

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=77, test_size = 0.25
)

import tensorflow as tf
tf.set_random_seed(777)

train_num = 4
test_num = 1
X = tf.placeholder(tf.float32, shape=[None, train_num])
Y = tf.placeholder(tf.int32, shape=[None, test_num])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot:",  Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:",  Y_one_hot)

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

def create_SoftMax_Layer(input_node, output_Node, layer_hypothesis, weight_name="weihgt", bias_name="bias"):
    W = tf.Variable(tf.random_normal([input_node, output_Node]), weight_name)
    b = tf.Variable(tf.random_normal([output_Node]), bias_name)
    logits = tf.matmul(X,W)+ b
    hypothesis = tf.nn.softmax(logits)
    return hypothesis, logits, W, b


Node_number = 16
hypothesis, logits, W, b = create_Relu_Layer(train_num,Node_number,X, "relu1")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu2")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu3")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,Node_number,hypothesis, "relu4")
hypothesis, logits, W, b = create_Relu_Layer(Node_number,train_num,hypothesis, "relu5")
hypothesis, logits, W, b = create_SoftMax_Layer(train_num, nb_classes,hypothesis)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, 
                                         labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        _, cost_val, acc_Val = sess.run(
            [optimizer, cost, accuracy], 
            feed_dict={X:x_train, Y: y_train}
        )
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tACC: {:.2%}".format(step, cost_val, acc_Val))
    
    pred = sess.run(prediction, feed_dict={X:x_test})
    for p , y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))
    
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess,
            feed_dict={X:x_test, Y:y_test}
        )
    )