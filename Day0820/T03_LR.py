import tensorflow as tf
tf.set_random_seed(777)
import numpy as np

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis -Y))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.train.AdagradDAOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

from sklearn.metrics import r2_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        cost_Val, hy_Val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict={X:x_data, Y: y_data}
        )
        print(step, "Cost: ", cost_Val,"\nPrediction:\n",hy_Val)
        r2_y_predict = r2_score(y_data, hy_Val)
        print("R2 :", r2_y_predict)
