import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
生成数据和标签
"""
x_data = np.linspace(0,1,500)[:,np.newaxis]
print(x_data.shape)
noise = np.abs(np.random.normal(0,0.05,x_data.shape))
y_data = np.square(x_data) + noise
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
pred_data = np.linspace(0,1,500)[:,np.newaxis]

"""
定义神经网络
"""
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
Bias_L1 = tf.Variable(tf.zeros([1,10]))
Bias_L2 = tf.Variable(tf.zeros([1,1]))
res_L1 = tf.matmul(x,Weight_L1) + Bias_L1
L1 = (tf.nn.sigmoid(res_L1))
res_L2 = tf.matmul(L1,Weight_L2) + Bias_L2
L2 = tf.nn.sigmoid(res_L2)

"""
计算损失函数
"""
loss = tf.reduce_mean(tf.square(y - L2))
print(type(loss))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

"""
进行训练及推理
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    pred_res = sess.run(L2, feed_dict={x: pred_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_res,'r-',lw=5)
    plt.show()
