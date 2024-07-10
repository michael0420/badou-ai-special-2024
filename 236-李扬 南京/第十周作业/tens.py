import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#随机生成数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

#定义placeholder存放数据
x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

#定义神经网络中间层
weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1,10]))
hid_out = tf.matmul(x, weight_L1) + biases_l1
L1 = tf.nn.tanh(hid_out)

#定义神经网络输出层
weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1,1]))
out_out = tf.matmul(L1, weight_L2) + biases_l2
L2 = tf.nn.tanh(out_out)

#损失函数
loss = tf.reduce_mean(tf.square(y - L2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    prediction = sess.run(L2, feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction, 'r-', lw=5)
    plt.show()