import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#构建-0.5~0.5等差数列，并且转换为(200, 1)二位数组
x_data = np.linspace(-0.5,0.5, 200)[:,np.newaxis]
noise = np.random.normal(0,0.01, x_data.shape)
y_data=np.square(x_data)+noise

#定义占位符，放入输入
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#构建中间层
w_l1 = tf.Variable(tf.random.normal([1,10]))
base_1 = tf.Variable(tf.zeros([1,10]))
w_pluse_l1 = tf.matmul(x, w_l1) + base_1 #加入偏置项
L1 = tf.nn.tanh(w_pluse_l1) #加入激活函数

#构建输出层
w_l2 = tf.Variable(tf.random.normal([10,1]))
base_2 = tf.Variable(tf.zeros([1,1]))
w_pluse_l2 = tf.matmul(L1, w_l2) + base_2
prediction = tf.nn.tanh(w_pluse_l2) #加入激活函数

#计算损失函数
loss = tf.reduce_mean(tf.square(y-prediction))

#反向传播，梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data,y:y_data})

    #预测
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r',lw=5) #画线
    plt.show()