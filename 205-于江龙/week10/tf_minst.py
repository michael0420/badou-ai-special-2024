import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate data
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# define two placeholder to feed data
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# define the neural network
weights_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1,10]))
wx_plus_b_l1 = tf.matmul(x,weights_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)

# define the output layer
weights_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros([1,1]))
wx_plus_b_l2 = tf.matmul(l1,weights_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)

# define the loss function
loss = tf.reduce_mean(tf.square(y-prediction))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()