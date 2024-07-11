import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

# 生成均匀分布的数据，并增加一个维度
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成正态分布的噪声数据，shape=x_data
noise = np.random.normal(0, 0.02, x_data.shape)
# y_data是x_data的平方加上noise
y_data = np.square(x_data) + noise

# 占位符，[None,1]表示第一个维度的大小不限，第二个维度是1
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 第一层的权重，初始化为正态分布，10个节点
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random.normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 计算预测值和实际标签值之间的均方误差，作为损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
