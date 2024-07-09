# _*_ coding: UTF-8 _*_
# @Time: 2024/6/27 19:23
# @Author: iris
# @Email: liuhw0225@126.com
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 生成300个随机点
    x_data = np.linspace(-0.8, 0.8, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.001, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义两个placeholder存放输入数据
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层
    weights_l1 = tf.Variable(tf.random_normal([1, 10]))
    biases_l1 = tf.Variable(tf.zeros([1, 10]))
    wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
    l1 = tf.nn.tanh(wx_plus_b_l1)

    weights_l2 = tf.Variable(tf.random_normal([10, 1]))
    biases_l2 = tf.Variable(tf.zeros([1, 1]))
    wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
    prediction = tf.nn.tanh(wx_plus_b_l2)

    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y - prediction))
    # 反向传播
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 训练2000次
        for i in range(20000):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # 获得预测值
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        # 画图
        plt.figure()
        plt.scatter(x_data, y_data)  # 散点是真实值
        plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
        plt.show()
