# tf实现神经网络""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成噪音点
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个变量存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
W1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))  # 偏置
prediction = tf.nn.relu(tf.matmul(x, W1) + b1)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练1000次
    for i in range(1000):
        # 训练
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
