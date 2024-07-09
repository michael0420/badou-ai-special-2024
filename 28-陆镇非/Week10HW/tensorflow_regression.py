import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-5,5,100).reshape(-1,1)
print(x_data.shape)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.sin(x_data)+noise
x_test = np.linspace(-5,5,100)[:,np.newaxis]
noise_test = np.random.normal(0,0.02,x_test.shape)
y_test = np.sin(x_test) + noise_test

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

wih = tf.Variable(tf.random.uniform([1,64],-1,1))
b1 = tf.Variable(tf.zeros([1,64]))
wih_x_plus_b = tf.matmul(x,wih)+b1
l1 = tf.nn.tanh(wih_x_plus_b) #[1,64]

who = tf.Variable(tf.random.uniform([64,1],-1,1))
b2 = tf.Variable(tf.zeros([1,1]))
who_x_plus_b = tf.matmul(l1,who)+b2
l2 = tf.nn.tanh(who_x_plus_b)
loss = tf.reduce_mean(tf.square(y-l2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction = sess.run(l2,feed_dict={x:x_data})

    plt.figure()
    plt.plot(x_data,y_data,label='truth')
    plt.plot(x_data,prediction,label='predict')
    plt.legend()
    plt.show()