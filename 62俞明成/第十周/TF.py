import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
matplotlib.use("TkAgg")

# 生成数据
# 生成一个包含200个元素的等差数列，将其转换为一个形状为(200, 1)的二维数组。
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 创建一个占位符（placeholder），用于输入数据。它的数据类型是浮点型（tf.float32）
# 形状为[None, 1]，其中None表示可以接受任意长度的的第一维数据
# 在TensorFlow程序运行时，需要通过feed_dict将具体的数据传入该占位符
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

weight_layer1 = tf.Variable(tf.random_normal([1, 10]))
bias_layer1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_layer1 = tf.matmul(x, weight_layer1) + bias_layer1
layer1 = tf.nn.tanh(wx_plus_b_layer1)

weight_layer2 = tf.Variable(tf.random_normal([10, 1]))
bias_layer2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_layer2 = tf.matmul(layer1, weight_layer2) + bias_layer2
prediction = tf.nn.tanh(wx_plus_b_layer2)

# 计算预测值与实际值之间的均方误差（MSE）,并将其累加，然后求平均值，返回最终的平均误差值。
# tf.square(y - prediction)用于计算预测值与实际值之间的差值的平方，tf.reduce_mean()用于计算平均值
loss = tf.reduce_mean(tf.square(y - prediction))

# 使用梯度下降优化器以学习率为0.1来最小化loss。
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 启动图后, 变量必须先经过`初始化`(init)初始化
    sess.run(tf.global_variables_initializer())
    # 训练5000次
    for i in range(5000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获取预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画出散点图和曲线
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()