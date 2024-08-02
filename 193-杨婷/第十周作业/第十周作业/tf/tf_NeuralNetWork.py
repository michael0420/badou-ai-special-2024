import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # [:, np.newaxis]将一维数组(-0.5, ..., 0.5)的形状从(200,)变为(200, 1)
noise = np.random.normal(0, 0.02, x_data.shape)  # mean:0 std:0.02
y_data = np.square(x_data) + noise
# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 定义神经网络中间层
wih = tf.Variable(tf.random.normal([1, 10]))  # 生成形状为(1,10)的均值0标准差1的随机权重
bias1 = tf.Variable(tf.zeros([1, 1]))  # 偏置
hidden_inputs = tf.matmul(x, wih) + bias1
'''
tf.nn提供了包括卷积、池化、归一化、损失函数、激活函数、分类操作、嵌入（Embedding）以及RNN（递归神经网络）等在内的多种神经网络构建所需的函数。
tf.nn.relu：ReLU激活函数，将输入值中小于0的部分置为0，保留大于0的部分。
tf.nn.sigmoid、tf.nn.tanh：Sigmoid和Tanh激活函数，分别将输入值映射到(0,1)和(-1,1)区间。
tf.nn.elu、tf.nn.softplus：其他非线性激活函数，用于增加网络的非线性能力。
'''
hidden_outputs = tf.nn.tanh(hidden_inputs)  # (?,10)

# 定义神经网络输出层
who = tf.Variable(tf.random.normal([10, 1]))
bias2 = tf.Variable(tf.zeros([1, 1]))
final_inputs = tf.matmul(hidden_outputs, who) + bias2
final_outputs = tf.nn.tanh(final_inputs)  # (?,1)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y-final_outputs))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # lr:0.1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(final_outputs, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)   # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)   # 曲线是预测值
    plt.show()





