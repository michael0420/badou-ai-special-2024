#!/usr/bin/env python
# coding: utf-8

# In[32]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#生成随机点
x_data =np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise =np.random.normal(0,0.02,x_data.shape)
y_data =np.square(x_data)+noise
# 定义两个palceholder存储有效数据
x =tf.placeholder(tf.float32,[None,1 
y =tf.placeholder(tf.float32,[None,1])

# 定义神经网络中间层，权重、偏置项、前向传播公式、加入激活函数
weights_L1 =tf.Variable(tf.random_normal([1,20]))
bias_L1 =tf.Variable(tf.zeros([1,20]))
Wx_plus_b_L1 =tf.matmul(x,weights_L1)+bias_L1
L1 =tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
weights_L2 =tf.Variable(tf.random_normal([20,1]))
bias_L2 =tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 =tf.matmul(L1,weights_L2)+bias_L2
pred =tf.nn.tanh(Wx_plus_b_L2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - pred))
# 反向传播算法
train_step =tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        sess.run(train_step, feed_dict ={x:x_data,y:y_data})
        
    prediction_value =sess.run(pred, feed_dict={x:x_data})
    
# 画图
plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,prediction_value,'r-',lw=5)
plt.show()



    

