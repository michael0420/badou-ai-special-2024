import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# # 1、随机生成200个[-0.5,0.5]的数据，并把它变成二维数组，即[200,1]的列向量
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# noise = np.random.normal(0, 0.02, x_data.shape)
# # y=x^2+noise
# y_data = np.square(x_data) + noise
#
# # 2、先用占位符声明x和y，[None,1]表示行数不确定，但是列数为1
# x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
# # 3、神经网络输入层到隐藏层
# # 定义输入层到隐藏层的随机权重
# # 这里的tf.random.normal必须要使用tf中的，不能使用np中的，否则会报错
# # weight_l1 = tf.Variable(np.random.normal([1, 10]))
# weight_l1 = tf.Variable(tf.random.normal([1, 10]))
# biases_l1 = tf.Variable(tf.zeros([1, 10]))
# # wx+b
# wx_plus_b1 = tf.matmul(x, weight_l1) + biases_l1
# # 过激活函数
# hidden_layers = tf.nn.tanh(wx_plus_b1)
#
# # 4、神经网络隐藏层到输出层
# weight_l2 = tf.Variable(tf.random.normal([10, 1]))
# biases_l2 = tf.Variable(tf.zeros([1, 1]))
# wx_plus_b2 = tf.matmul(hidden_layers, weight_l2) + biases_l2
# # 过激活函数
# output_layers = tf.nn.tanh(wx_plus_b2)
#
# # 5、定义损失函数，均值平方差
# loss = tf.reduce_mean(tf.square(y - output_layers))
# # 定义反向传播算法
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# # 6、真正训练
# with tf.Session() as sess:
#     # 初始化全局变量
#     sess.run(tf.global_variables_initializer())
#     # 训练2000次
#     for i in range(2000):
#         # 训练
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#     # 获取预测值
#     prediction_val = sess.run(output_layers, feed_dict={x: x_data})
#
# # 6、画图
# plt.figure()
# # 画(x_data,y_data)对应的散点图
# plt.scatter(x_data, y_data)
# # 'r-'表示红色，lw表示线宽
# plt.plot(x_data, prediction_val, 'r-', lw=5)
# plt.show()
#
#
# # 1、生成随机数据
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# noise = np.random.normal(0, 0.02, x_data.shape)
# y_data = np.square(x_data) + noise
# # 2、占位符定义x和y
# x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# # 3、输入层到隐藏层
# weight_l1 = tf.Variable(tf.random.normal([1, 10]))
# biases_l1 = tf.Variable(tf.zeros([1, 10]))
# wx_plus_b1 = tf.matmul(x, weight_l1) + biases_l1
# # 过激活函数
# hidden_layers = tf.tanh(wx_plus_b1)
# # 4、隐藏层到输出层
# weight_l2 = tf.Variable(tf.random_normal([10, 1]))
# biases_l2 = tf.Variable(tf.zeros([1, 1]))
# wx_plus_b2 = tf.matmul(hidden_layers, weight_l2) + biases_l2
# # 过激活函数
# output_layers = tf.tanh(wx_plus_b2)
# # 5、计算损失函数
# loss = tf.reduce_mean(tf.square(y - output_layers))
# # 6、通过反向传播调整权重
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# # 7、训练
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2000):
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#     prediction_val = sess.run(output_layers, feed_dict={x: x_data})
# # 8、画图
# plt.figure()
# plt.scatter(x_data, y_data)
# plt.plot(x_data, prediction_val, 'r-', lw=5)
# plt.show()
#
#
# # 1、生成随机数据
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# noise = np.random.normal(0, 0.01, x_data.shape)
# y_data = np.square(x_data) + noise
# # 2、定义x和y占位
# x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# # 3、输入层到隐藏层
# # 随机生成权重矩阵
# weight_l1 = tf.Variable(tf.random_normal([1, 10]))
# biases_l1 = tf.Variable(tf.zeros([1, 10]))
# wx_plus_b1 = tf.matmul(x, weight_l1) + biases_l1
# # 过激活函数
# hidden_layers = tf.tanh(wx_plus_b1)
# # 4、隐藏层到输出层
# weight_l2 = tf.Variable(tf.random_normal([10, 1]))
# biases_l2 = tf.Variable(tf.zeros([1, 1]))
# wx_plus_b2 = tf.matmul(hidden_layers, weight_l2) + biases_l2
# # 过激活函数
# output_layers = tf.tanh(wx_plus_b2)
# # 求损失函数
# loss = tf.reduce_mean(tf.square(y - output_layers))
# # 5、反向传播
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# # 训练
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2000):
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#     predict_val = sess.run(output_layers, feed_dict={x: x_data})
# plt.figure()
# plt.scatter(x_data, y_data)
# plt.plot(x_data, predict_val, 'r-', lw=5)
# plt.show()


# # 1、生成随机数据
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# noise = np.random.normal(0, 0.01, x_data.shape)
# y_data = np.square(x_data) + noise
# # 2、定义x和y的占位符
# x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# # 3、定义神经网络
# weight_l1 = tf.Variable(tf.random_normal([1, 10]))
# biases_l1 = tf.Variable(tf.zeros([1, 10]))
# wx_plus_b1 = tf.matmul(x, weight_l1) + biases_l1
# hidden_layers = tf.tanh(wx_plus_b1)
#
# weight_l2 = tf.Variable(tf.random_normal([10, 1]))
# biases_l2 = tf.Variable(tf.zeros([1, 1]))
# wx_plus_b2 = tf.matmul(hidden_layers, weight_l2) + biases_l2
# output_layers = tf.tanh(wx_plus_b2)
# # 4、损失函数（均方差）
# loss = tf.reduce_mean(tf.square(y - output_layers))
# # 5、误差反向传播
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# # 6、训练
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2000):
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#     prediction_val = sess.run(output_layers, feed_dict={x: x_data})
# # 7、画图
# plt.figure()
# plt.scatter(x_data, y_data)
# plt.plot(x_data, prediction_val, 'r-', lw=5)
# plt.show()
#


# 随机生成200个数据
x_data = np.linspace(-0.5, 0.5, num=200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, size=(200, 1)) - 0.5
y_data = np.square(x_data) + noise
# plt.figure()
# plt.scatter(x_data,y_data)
# plt.show()

x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 构建神经网络
# 输入层到隐藏层
weight_l1 = tf.Variable(initial_value=tf.random.normal(shape=(1, 10), dtype=tf.float32))
bias_1 = tf.Variable(tf.zeros(shape=(1, 10), dtype=tf.float32))
wx_plus_b1 = tf.matmul(x, weight_l1) + bias_1
L1 = tf.nn.tanh(wx_plus_b1)

# 隐藏层到输出层
weight_l2 = tf.Variable(initial_value=tf.random_normal(shape=(10, 1), dtype=tf.float32))
bias_2 = tf.Variable(tf.zeros(shape=(1, 1), dtype=tf.float32))
wx_plus_b2 = tf.matmul(L1, weight_l2) + bias_2
L2 = tf.nn.tanh(wx_plus_b2)

# 损失函数
mse = tf.keras.losses.MeanSquaredError()
loss = mse(y, L2)

# 反向传播
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

epochs = 3000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        sess.run(fetches=train, feed_dict={x: x_data, y: y_data})
    y_pred = sess.run(fetches=L2, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, y_pred, 'r-', lw=5)
    plt.show()
