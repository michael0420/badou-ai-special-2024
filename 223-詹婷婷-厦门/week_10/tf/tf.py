import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
输入
"""
#使用numpy随机生成200个点
#生成200行1列的二维数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder存放数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

"""
隐藏层
"""
#随机初始化权重
weights_L1 = tf.Variable(tf.random_normal([1,10]))#服从 “正态分布” 中生成随机数
biases_L1 = tf.Variable(tf.zeros([1,10]))#生成全0的tensor张量
wx_plus_b_l1 = tf.matmul(x, weights_L1) + biases_L1
l1 = tf.nn.tanh(wx_plus_b_l1)

"""
输出层
"""
#随机初始化权重
weights_L2 = tf.Variable(tf.random_normal([10,1]))#服从 “正态分布” 中生成随机数
biases_L2 = tf.Variable(tf.zeros([1,1]))#生成全0的tensor张量
wx_plus_b_l2 = tf.matmul(l1, weights_L2) + biases_L2
predict = tf.nn.tanh(wx_plus_b_l2)

#定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y-predict))

#定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    predict_res = sess.run(predict, feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predict_res, 'r-', lw=5)
    plt.show()