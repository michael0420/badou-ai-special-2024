import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# 定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

# 定义神经网络中间层
w1=tf.Variable(tf.random.normal([1,10]))
# 加入偏置项
b1=tf.Variable(tf.zeros([1,10]))
l1=tf.matmul(x,w1)+b1
# 加入激活函数
activation=tf.nn.tanh(l1)

# 定义神经网络输出层
w2=tf.Variable(tf.random.normal([10,1]))
# 加入偏置项
b2=tf.Variable(tf.zeros([10,1]))
l2=tf.matmul(l1,w2)+b2
# 加入激活函数
prediction=tf.nn.tanh(l2)

# 定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(y-prediction))

# 定义反向传播算法（使用梯度下降算法训练）
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练5000次
    for i in range(5000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
