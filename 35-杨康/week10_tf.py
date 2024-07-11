import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#设置数据集
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]  #200行1列
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

#设置占位符
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#设置中间层
W1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,W1) + b1
L1 = tf.nn.tanh(Wx_plus_b_L1)  #激活函数tanh

#设置输出层
W2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,W2) + b2
prediction = tf.nn.tanh(Wx_plus_b_L2)  #激活函数tanh

#设置损失函数 均方误差
loss = tf.reduce_mean(tf.square(y-prediction))
#设置反向传播方法  梯度下降法
trainstep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(trainstep,feed_dict ={x:x_data,y:y_data})

    #进行预测
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r',lw=5)
    plt.show()


