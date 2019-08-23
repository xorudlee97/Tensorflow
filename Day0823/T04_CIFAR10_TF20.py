import tensorflow as tf
from keras.datasets import cifar10
import numpy as np;
import random;

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# def build_CNN_classifier(X_img):
#     W = tf.Variable(tf.random_normal([5, 5, 3, 64],stddev=5e-2))
#     Model_Layer = tf.nn.relu(tf.nn.conv2d(X_img, W, strides=[1, 1, 1, 1], padding='SAME'))
#     Model_Layer = tf.nn.max_pool(Model_Layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")

#     W = tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=5e-2))
#     Model_Layer = tf.nn.relu(tf.nn.conv2d(Model_Layer, W, strides=[1, 1, 1, 1], padding='SAME'))
#     Model_Layer = tf.nn.max_pool(Model_Layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#     W = tf.Variable(tf.random_normal([3, 3, 64, 128],stddev=5e-2))
#     Model_Layer = tf.nn.relu(tf.nn.conv2d(Model_Layer, W, strides=[1, 1, 1, 1], padding='SAME'))

#     W = tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=5e-2))
#     Model_Layer = tf.nn.relu(tf.nn.conv2d(Model_Layer, W, strides=[1, 1, 1, 1], padding='SAME'))

#     W = tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=5e-2))
#     Model_Layer = tf.nn.relu(tf.nn.conv2d(Model_Layer, W, strides=[1, 1, 1, 1], padding='SAME'))

#     W = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
#     Model_flat = tf.reshape(Model_Layer, [-1, 8*8*128])
#     Model_Layer = tf.nn.relu(tf.matmul(Model_flat, W))


#     Model_Layer = tf.nn.dropout(Model_Layer, keep_prob=0.8)

#     W = tf.get_variable("W3", shape=[384, 10], initializer=tf.contrib.layers.xavier_initializer())
#     b =  tf.Variable(tf.random_normal([10]))
#     model = tf.matmul(Model_Layer, W) + b
#     y_pred = tf.nn.softmax(model)

#     return model, y_pred
def build_CNN_classifier(X_img):
    Model_Layer = tf.layers.conv2d(X_img, 64, [5, 5], activation=tf.nn.relu, padding= "same")
    Model_Layer = tf.layers.max_pooling2d(Model_Layer, [3,3], [2,2], padding="same")

    Model_Layer = tf.layers.conv2d(Model_Layer, 64, [5, 5], activation=tf.nn.relu, padding= "same")
    Model_Layer = tf.layers.max_pooling2d(Model_Layer, [3,3], [2,2], padding= "same")

    Model_Layer = tf.layers.conv2d(Model_Layer, 64, [3, 3], activation=tf.nn.relu, padding= "same")
    Model_Layer = tf.layers.max_pooling2d(Model_Layer, [3,3], [2,2], padding= "same")

    Model_Layer = tf.layers.conv2d(Model_Layer, 128, [3, 3], activation=tf.nn.relu, padding= "same")
    Model_Layer = tf.layers.conv2d(Model_Layer, 128, [3, 3], activation=tf.nn.relu, padding= "same")
    Model_Layer = tf.layers.conv2d(Model_Layer, 128, [3, 3], activation=tf.nn.relu, padding= "same")

    Model_Layer = tf.contrib.layers.flatten(Model_Layer)
    Model_Layer = tf.layers.dense(Model_Layer, 384, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, 256, activation=tf.nn.relu)
    Model_Layer = tf.layers.dense(Model_Layer, 128, activation=tf.nn.relu)
    Model_Layer = tf.layers.dropout(Model_Layer, 0.8)

    model = tf.layers.dense(Model_Layer, 10, activation=None)

    return model

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32


X = tf.placeholder(tf.float32, [None, IMG_ROWS, IMG_COLS, IMG_CHANNELS])
X_img = tf.reshape(X, [-1, IMG_COLS, IMG_ROWS, IMG_CHANNELS])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, 10)
print("one_hot:",  Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 10])
print("reshape one_hot:",  Y_one_hot)

model = build_CNN_classifier(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Learning started. It takes sometime.")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X: batch_xs, Y: batch_ys}
            cost_Val, _ = sess.run([cost, optimizer], feed_dict= feed_dict)
            avg_cost += cost_Val /total_batch
        
        print("Epoch: %04d" %(epoch +1), "cost = {:.9f}" .format(avg_cost))
    
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", sess.run(accuracy, feed_dict={
        X: x_test, Y : y_test
    }))

    re = random.randint(0, len(y_test) -1)
    print("Label:", sess.run(tf.argmax(y_test[re:re+1], 1)))
    print("Prediction", sess.run(tf.argmax(model, 1), feed_dict = {X: x_test[re:re+1]}))