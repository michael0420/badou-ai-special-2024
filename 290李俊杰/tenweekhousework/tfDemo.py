'''
第十周作业
1.使用keras实现简单神经网络
2.用代码从零实现推理过程
3.使用tf实现简单神经网络

'''

# 3.使用tf实现简单神经网络
import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#使用numpy生成150个随机点
x_data=np.linspace(0,1,150)[:,np.newaxis]
noise=np.random.normal(0,0.03,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder存放输入数据
xp=tf.placeholder(tf.float32,[None,1])
yp=tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
w_l1=tf.Variable(tf.random.normal([1,10]))
b_l1=tf.Variable(tf.zeros([1,10]))
wxb_l1=tf.matmul(xp,w_l1)+b_l1
# 过激活函数
l1=tf.nn.tanh(wxb_l1)

#定义神经网络输出层
w_l2=tf.Variable(tf.random.normal([10,1]))
b_l2=tf.Variable(tf.zeros([1,1]))
wxb_l2=tf.matmul(l1,w_l2)+b_l2
l2=tf.nn.tanh(wxb_l2)

#定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(yp-l2))

#定义反向传播算法（使用梯度下降算法训练）
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 创建会话
with tf.Session() as tfse:
    # 初始化变量
    tfse.run(tf.global_variables_initializer())
    #训练2500次
    for i in range(2500):
        tfse.run(train,feed_dict={xp:x_data,yp:y_data})

    #获得预测值
    estimate=tfse.run(l2,feed_dict={xp:x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, estimate, 'r-', lw=5)  # 曲线是预测值
    plt.show()



