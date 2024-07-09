import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

o_data = np.linspace(-0.5, 0.5, 300)[:, np.newaxis]
gaoshizaoshen = np.random.normal(0, 0.02, o_data.shape)
k_data = np.square(o_data) + gaoshizaoshen
# 定义两个变量存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 随机设置权重
weight = tf.Variable(tf.random_normal([1, 10]))
balse = tf.Variable(tf.zeros([1, 1]))
zinput = tf.matmul(x, weight) + balse
aout = tf.nn.tanh(zinput)

weight1 = tf.Variable(tf.random_normal([10, 1]))
balse1 = tf.Variable(tf.zeros([1, 1]))
zinput1 = tf.matmul(aout, weight1) + balse1
aout1 = tf.nn.tanh(zinput1)

# 输出层
weight2 = tf.Variable(tf.random_normal([1, 10]))
balsetwo = tf.Variable(tf.zeros([1, 1]))
zinput2 = tf.matmul(aout1, weight2) + balsetwo
aout2 = tf.nn.tanh(zinput2)
# 定义损失函数
loss = tf.reduce_mean(tf.square(y - aout2))
# 定义反向传播算法
minimize = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(minimize, feed_dict={x: o_data, y: k_data})

    # 获得预测值
    prediction_value = sess.run(aout2, feed_dict={x: o_data})

    # 画图
    plt.figure()
    plt.scatter(o_data, k_data)  # 散点是真实值
    plt.plot(o_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
