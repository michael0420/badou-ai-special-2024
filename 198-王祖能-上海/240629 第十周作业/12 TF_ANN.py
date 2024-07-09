import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 300)[:, np.newaxis]  # linspace均布产生-1~1之间的300个数，并将一维数组转化为二维矩阵，增加维度且按列排
noise = np.random.normal(0, 0.03, X.shape)
Y = np.exp(X) + noise
# 可能用到的变量先占位
input = tf.placeholder(tf.float32, [None, 1])
output = tf.placeholder(tf.float32, [None, 1])
# 需要显式写出神经网络结构
w1 = tf.Variable(tf.random_normal(shape=[1, 8], mean=0, stddev=5.))  # random和zeros默认都是float32类型
b1 = tf.Variable(tf.zeros([1, 8]))
sum_weight1 = tf.matmul(input, w1) + b1  # 如果按照w1 * x + b那么w1的shape就是[300, 10]，数据越多参数越多
Y_hide = tf.nn.tanh(sum_weight1)

w2 = tf.Variable(tf.random_normal(shape=[8, 1], mean=0, stddev=5.))
b2 = tf.Variable(tf.zeros([1, 1]))
sum_weight2 = tf.matmul(Y_hide, w2) + b2
Y_pred = tf.nn.tanh(sum_weight2)

loss = tf.reduce_mean(tf.square(Y_pred - output))  # 方差均值，加标签定义是否降维度输出
train_model = tf.train.GradientDescentOptimizer(0.1)  # lr=0.1
train_model = train_model.minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())  # 变量初始化不要忘记括号
  for i in range(800):  # keras封装后简单只需要nn.fit参数包含了epochs, batchsize。而不需要for loop
    sess.run([train_model], feed_dict={input: X, output: Y})
    cost = sess.run(loss, feed_dict={input: X, output: Y})
    # plt.scatter(i, cost)
  prediction = sess.run(Y_pred, feed_dict={input: X})

  plt.scatter(X, Y)
  plt.plot(X, prediction, c='purple', lw=5)
  plt.show()
