import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) +noise

#定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#hidden
weight_L1 = tf.Variable(tf.random_normal([1,10]))
baises_L1 = tf.Variable(tf.zeros([1,10]))
WX_plus_b = tf.matmul(x,weight_L1)+baises_L1
L1 = tf.nn.tanh(WX_plus_b)
#output
weight_L2 = tf.Variable(tf.random_normal([10,1]))
baises_L2 = tf.Variable(tf.zeros([1,1]))
WX_plus_b2 = tf.matmul(L1,weight_L2)+baises_L2
L2 = tf.nn.tanh(WX_plus_b2)

#loss
loss = tf.reduce_mean(tf.square(y-L2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_value = sess.run(L2,feed_dict={x:x_data})


    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-')
    plt.show()


















