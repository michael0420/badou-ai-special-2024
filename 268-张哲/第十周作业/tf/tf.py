import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
#np.linspace:生成从-0.5到0.5的200个等间隔数值
# x[:, np.newaxis] ：放在后面，会给列上增加维度；
# x[np.newaxis, :] ：放在前面，会给行上增加维度；
x_data = np.linspace(-0.5,0.5,200)
print(x_data.shape)
x_data = x_data[:,np.newaxis]
print(x_data.shape)
#生成均值为0，方差为0.02的正态分布的随机数，size是输出的shape
noise = np.random.normal(0,0.02,size=x_data.shape)
print(noise.shape)
y_data=np.square(x_data)+noise
#定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
print(x.shape)
#定义神经网络中间层
w_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1,10])) #偏置项
wx_plus_b_l1 = tf.matmul(x,w_l1)+biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1) #激活函数
#定义神经网络输出层
w_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros([1,1]))
wx_plus_b_l2 = tf.matmul(l1,w_l2)+biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)
#定义损失函数（均方差函数
loss = tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# 开始训练
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 预测
    prediction_val = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_val,'g-',lw=5)#绿色实线 宽度为5
    plt.show()

