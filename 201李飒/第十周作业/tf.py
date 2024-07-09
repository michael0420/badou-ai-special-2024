import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 使用numpy生成200个随机数，
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 用tensorflow开始构建网络
# 定义两个placeholder来 储存输入的数据 ，placeholder作为占位符
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义神经网络的中间层 ,把权重设定为一个变量，初始值是
Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
zh1 =tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(zh1)

# 定义神经网络的输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
zo1 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(zo1)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 反向传播
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data,y:y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()





