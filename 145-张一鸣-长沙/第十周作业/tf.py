# coding = utf-8

'''
    用tensorflow实现训练
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成数据
X = np.linspace(-0.5, 0.5, 250)[:, np.newaxis]
# np.linspace生成一个从-0.5到0.5的等差数列，总共有200个点，生成一个一维数组
# [:, np.newaxis]将一维数组转换为二维数组，使一维数组（形状为(200,)）变为二维数组（形状为(200,1)）
noise = np.random.normal(0, 0.02, X.shape)
# 生成与X形状相同的随机噪声,np.random.normal(0,0.02,X.shape)从均值为0、标准差为0.02的正态分布中抽取随机数，生成一个形状为(200,1)的二维数组
Y = np.square(X) + noise

# 定义placeholder占位
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构建隐藏层
w_ih = tf.Variable(tf.random_normal([1, 10]))
b_ih = tf.Variable(tf.zeros([1, 10]))
zh = tf.matmul(x, w_ih) + b_ih      # wx + b
ho = tf.nn.tanh(zh)      # 激活函数

# 构建输出层
w_ho = tf.Variable(tf.random_normal([10, 1]))
b_ho = tf.Variable(tf.zeros([1, 1]))
zo = tf.matmul(ho, w_ho) + b_ho
oo = tf.nn.tanh(zo)

# 定义损失函数（均方差）
loss = tf.reduce_mean(tf.square(y - oo))
# 定义梯度下降法更新权重
gdd = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.Session() as ss:
    # Variab初始化 & Session.run()
    ss.run(tf.global_variables_initializer())
    # 训练2500次，需feed数据样本与标签反向传播更新权重
    for i in range(2500):
        ss.run(gdd, feed_dict={x: X, y: Y})
    # 推理（实际数据与训练/测试集不同），传入执行图，仅feed数据
    predict = ss.run(oo, feed_dict={x: X})

# plt绘图
plt.figure()        # 创建默认图形窗口
plt.scatter(X, Y)       # 原数据散点图
plt.plot(X, predict, 'r-', lw=5)    # 绘制线条，'r'-红色red，'-'-实线，线条宽度linewidth为5
plt.show()


