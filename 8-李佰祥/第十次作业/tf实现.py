import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#定义placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
weights_L1 = tf.Variable(tf.random.normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b =  tf.matmul(x,weights_L1) + biases_L1
#加入激活函数
L1 = tf.nn.relu(Wx_plus_b)


#定义输出层
Wwight_L2 = tf.Variable(tf.random.normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b = tf.matmul(L1,Wwight_L2) +biases_L2
prediction = tf.nn.relu(Wx_plus_b)


#定义损失函数(均方差函数)
loss = tf.reduce_mean(tf.square(y-prediction))

#反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_values = sess.run(prediction,feed_dict={x:x_data})








