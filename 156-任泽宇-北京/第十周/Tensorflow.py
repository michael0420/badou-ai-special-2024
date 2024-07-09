import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成输入蹭
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# print(x_data.shape)
# 生成200*1的均值为0，标准差为0.02的数
noise = np.random.normal(0, 0.02, x_data.shape)
# 生成输出层
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(np.float32, [None, 1])
y = tf.placeholder(np.float32, [None, 1])

# 定义网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
# 加入激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)


# 定义网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练次数
    for i in range(3000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()