# -*- coding: utf-8 -*-
"""
@File    :   tf.py
@Time    :   2024/06/30 16:01:16
@Author  :   廖红洋 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-1,1,200)[:,np.newaxis]#线性X序列
noise = np.random.normal(0,0.08,x_data.shape)
y_data = np.square(x_data)+noise

# tensorflow 1.x版本中的placeholder，在tf2中已经被取消，在tf2中，执行为动态图，
#而非静态图，这代表再使用placeholder和session执行已经不行，需要关闭eager_execution
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32,[None,1])#第二个参数是维度，表示不定行数，一列的数组
y = tf.compat.v1.placeholder(tf.float32,[None,1])

# 定义网络
L1_weights = tf.Variable(tf.random.normal([1, 10])) # 设定初始值的变量权重矩阵
L1_bias = tf.Variable(tf.zeros([1,10]))
L1_val = tf.matmul(x,L1_weights)+L1_bias # 计算机矩阵乘法是否自检？是否需要
L1_out = tf.nn.tanh(L1_val)
L2_weights = tf.Variable(tf.random.normal([10, 1])) # 只输出一个值，因此为10行一列
L2_bias = tf.Variable(tf.zeros([1,1]))
L2_val = tf.matmul(L1_out,L2_weights)+L2_bias
L2_out = tf.nn.tanh(L2_val)

# 损失函数
loss = tf.reduce_mean(tf.square(y - L2_out))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

# 训练
epochs = 4000
with tf.compat.v1.Session() as sess: # 加强版的异常处理，with as语法
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(epochs): # 轮次200
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 获得预测值
    prediction = sess.run(L2_out, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction, "r-", lw=5)  # 曲线是预测值
    plt.show()