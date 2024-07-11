import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
# [:, np.newaxis] 将一维数组变成二维数组，每个点作为一个独立的行。最终 x_data 的形状为 (200, 1)。
#np.linspace(-0.5,0.5,200):随机生成均匀分布在-0.5~0.5范围内的200个点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
x_test = np.linspace(-0.5,0.5,50)[:,np.newaxis]
noise_test = np.random.normal(0,0.02,x_test.shape)
y_test = np.square(x_test) + noise_test

#定义数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
wih = tf.Variable(tf.random.normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
#计算
wih_x_plus_b = tf.matmul(x,wih)+b1
#激活函数输出
l1 = tf.nn.tanh(wih_x_plus_b) #[1,10]

#定义神经网络输层
who = tf.Variable(tf.random.normal([10,1]))
# tf.matmul(l1,who) => [1,1]
b2 = tf.Variable(tf.zeros([1,1]))
who_x_plus_b = tf.matmul(l1,who)+b2
l2 = tf.nn.tanh(who_x_plus_b)
# 定义损失函数
loss = tf.reduce_mean(tf.square(y-l2))
#定义反向传播函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction = sess.run(l2,feed_dict={x:x_test})

    #画图
    plt.figure()
    plt.scatter(x_test,y_test)
    # plt.scatter(x_data, y_data)
    plt.plot(x_test,prediction,'r-',lw=5)
    plt.show()
