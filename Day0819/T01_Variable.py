import tensorflow as tf
tf.set_random_seed(777)

weight = tf.Variable(tf.random_normal([1]), name="weight")
bias = tf.Variable(tf.random_normal([1]), name="bias")

print(weight)

weight =  tf.Variable([0.3], tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    weight_eval = weight.eval(session=sess)
    print(weight_eval)
    print(sess.run(bias))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
weight_eval = weight.eval()
print(weight_eval)
sess.close()