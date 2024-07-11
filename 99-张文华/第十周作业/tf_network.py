'''
使用tf实现啊简单的神经网络
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 生成随机点
x_data = np.linspace(-1, 1, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义占位
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义模型结构
# 第一层
w1 = tf.Variable(tf.random.normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
zh1 = tf.matmul(x, w1) + b1
ah1 = tf.nn.tanh(zh1)

# 第二层
w2 = tf.Variable(tf.random.normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
zh2 = tf.matmul(ah1, w2) + b2
ah2 = tf.nn.tanh(zh2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - ah2))

# 定义反向传播算法，使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 执行训练
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练两千次
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    # 获取预测值
    pre_data = sess.run(ah2, feed_dict={x:x_data})

# 画图
plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_data, pre_data, 'r-', lw=10)
plt.show()



