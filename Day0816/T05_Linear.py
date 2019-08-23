import tensorflow as tf
tf.set_random_seed(777)

# 데이터
x_train = [1,2,3]
y_train = [1,2,3]

# 모델 생성
weight = tf.Variable(tf.random_normal([1]), name= 'weight')
bias = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesls = x_train * weight + bias

# model.complie
# cost / loss
cost = tf.reduce_mean(tf.square(hypothesls - y_train))
# Optimizer : Learninglate (중요)
# ex) Adam
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 세션 시작
with tf.Session() as sess:
    # 변수 생성 시 초기화를 해야 하기 때문에 쓰인다.
    sess.run(tf.global_variables_initializer())

    # model.fit
    #훈련 횟수 2000번, sess.run([train, cost, weight, bias]) == model.fit(x_trian, y_train)
    for step in range(2001):
        _, cost_val, Weight_val, Bias_val = sess.run([train, cost, weight, bias])

        if step % 20 ==0:
            print(step, cost_val, Weight_val, Bias_val)