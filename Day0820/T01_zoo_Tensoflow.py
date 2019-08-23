import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
import os
tf.set_random_seed(777)

file_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Data/")
xy_data = np.loadtxt(file_dir+"/data-04-zoo.csv", delimiter=",", dtype=np.float32)

x_data = xy_data[:,:-1]
y_data = xy_data[:,[-1]]
nb_classes = 7

y_data = to_categorical(y_data)

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])


W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X,W)+ b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

correct_predict = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val, _ = sess.run(
            [cost, optimizer], 
            feed_dict={X:x_data, Y: y_data}
        )
        if step % 200 == 0:
            print(step, cost_val)
    
    hy_Val, pre_Val, acc = sess.run(
        [hypothesis, correct_predict, accuracy],
        feed_dict={X:x_data, Y:y_data}
    )
    print(
        "\nHypothesis:\n", hy_Val,
        "\nCorrect (Y):", pre_Val,
        "\nAccuracy:", acc
    )
'''
# https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data,,,,,,,,,,,,,,,,
#   1. animal name:     (deleted),,,,,,,,,,,,,,,,
#   2. hair     Boolean",,,,,,,,,,,,,,,,
#   3. feathers     Boolean",,,,,,,,,,,,,,,,
#   4. eggs     Boolean",,,,,,,,,,,,,,,,
#   5. milk     Boolean",,,,,,,,,,,,,,,,
#   6. airborne     Boolean",,,,,,,,,,,,,,,,
#   7. aquatic      Boolean",,,,,,,,,,,,,,,,
#   8. predator     Boolean",,,,,,,,,,,,,,,,
#   9. toothed      Boolean",,,,,,,,,,,,,,,,
#  10. backbone     Boolean",,,,,,,,,,,,,,,,
#  11. breathes     Boolean",,,,,,,,,,,,,,,,
#  12. venomous     Boolean",,,,,,,,,,,,,,,,
#  13. fins     Boolean",,,,,,,,,,,,,,,,
#  14. legs     Numeric (set of values: {0",2,4,5,6,8}),,,,,,,,,,,
#  15. tail     Boolean",,,,,,,,,,,,,,,,
#  16. domestic     Boolean",,,,,,,,,,,,,,,,
#  17. catsize      Boolean",,,,,,,,,,,,,,,,
#  18. type     Numeric (integer values in range [0",6]),,,,,,,,,,,,,,,
'''