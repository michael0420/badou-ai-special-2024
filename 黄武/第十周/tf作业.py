'''
使用TensorFlow框架实现一个简单的神经网络模型，
用于拟合一个带有噪声的二次函数。
首先生成200个随机点作为训练数据，然后定义神经网络的结构，包括输入层、中间层和输出层。
接着定义损失函数（均方差函数）和反向传播算法（使用梯度下降算法训练）。
最后在会话中初始化变量，进行2000次训练，并绘制散点图和预测曲线。
'''


import tensorflow as tf   #用于构建和训练神经网络
import numpy as np        #用于数值计算
import matplotlib.pyplot as plt  #用于数据可视化

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]  #随机生成200个数
noise = np.random.normal(0,0.02,x_data.shape)  # 高斯噪声，正态分布，与x_data形状相同
y_data = np.square(x_data) + noise
'''
创建x和y占位符  placeholder
数据类型是 tf.float32,
[None,1],一个二维数组，其中第一维的大小是任意的，第二维的大小是 1，
意味着每个样本是一个具有一个元素的一维数组。
'''
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

'''
创建中间层的权重Weights_L1和偏置biases_L1,并对输入数据x应用线性变换和激活函数tanh
调用生成一个形状为 [1,10] 的随机数数组，
数组中的数值来自标准正态分布（均值为 0，方差为 1）
函数调用将生成的随机数数组转换为一个 TensorFlow 变量。
在 TensorFlow 中，变量是可以在训练过程中更新的值。
权重是模型训练过程中需要学习的参数，因此它们被定义为变量。
'''
Weights_L1=tf.Variable(tf.random_normal([1,10]))

# 函数调用生成一个形状为 [1,10] 的数组，数组中的所有元素都初始化为 0。偏置

biases_L1 = tf.Variable(tf.zeros([1,10]))

#将权重矩阵 W 与输入数据 x 进行矩阵乘法，然后加上偏置项 b。
Wx_plus_b_L1=tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  #激活函数

#创建神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y-prediction))

#定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

  #画图
    plt.figure()
    plt.scatter(x_data,y_data)   #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
    plt.show()
