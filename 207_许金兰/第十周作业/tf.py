"""
@author: 207-xujinlan
使用tensorflow构建神经网络
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置输入数据和标签数据变量
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
# 神经网络隐藏层计算
w1=tf.Variable(tf.random_normal([1,10]))   # 随机初始化隐藏层参数
# b1=tf.Variable(tf.random_normal([1,10]))  # 增加隐藏层偏置项
b1=tf.Variable(tf.zeros([1,10]))
hidden_z=tf.matmul(x,w1)+b1   # 隐藏层输入
hidden_o=tf.nn.tanh(hidden_z)  # 隐藏层激活
# 神经网络输出层计算
w2=tf.Variable(tf.random_normal([10,1]))
# b2=tf.Variable(tf.random_normal([1,1]))
b2=tf.Variable(tf.zeros([1,1]))
out_z=tf.matmul(hidden_o,w2)+b2
prediction=tf.nn.tanh(out_z)

# 设置损失函数
loss=tf.reduce_mean(tf.square(y-prediction))
# 模型训练
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 数据构造
x_data=np.linspace(-1,1,200)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+noise
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练模型
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 模型预测
    predict_result=sess.run(prediction,feed_dict={x:x_data})
    # 绘图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,predict_result,'r-',lw=5)
    plt.show()

