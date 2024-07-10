import tensorflow as tf
import numpy as np
import Cifar10_read_data_homework as read_data
import math

epochs = 4000
batch_size = 100
test_num = 10000
data_dir="Cifar_data/cifar-10-batches-bin"

def weight_variable(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var

train_images, train_labels = read_data.input_data(distorted=True, batch_size=batch_size, data_dir=data_dir)
test_images, test_labels = read_data.input_data(distorted=False, batch_size=batch_size, data_dir=data_dir)
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])

"""
第一个卷积层
"""
kernel1 = weight_variable([5, 5, 3, 64], stddev=0.05, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
input1 = tf.nn.bias_add(conv1, bias1)
relu1 = tf.nn.relu(input1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

"""
第二个卷积层
"""
kernel2 = weight_variable([5, 5, 64, 64],stddev=0.05,w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
input2 = tf.nn.bias_add(conv2, bias2)
relu2 = tf.nn.relu(input2)
pool2 = tf.nn.max_pool(input2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

data_reshape = tf.reshape(pool2, [batch_size, -1])
dim = data_reshape.get_shape()[1].value

"""
第一个全连接层
"""
weight1 = weight_variable([dim, 384],stddev=0.04,w1=0.004)
fc_b1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_out1 = tf.nn.relu(tf.matmul(data_reshape, weight1) + fc_b1)

"""
第二个全连接层
"""
weight2 = weight_variable([384, 192], stddev=0.04, w1=0.004)
fc_b2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_out2 = tf.nn.relu(tf.matmul(fc_out1, weight2) + fc_b2)

"""
第三个全连接层
"""
weight3 = weight_variable([192, 10], stddev=1 / 192.0, w1=0.0)
fc_b3 = tf.Variable(tf.constant(0.1, shape=[10]))
fc_out3 = tf.matmul(fc_out2, weight3) + fc_b3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_out3, labels=tf.cast(y, tf.int32))
weight_L2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weight_L2_loss
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

top_k = tf.nn.in_top_k(fc_out3, y, 1)

option = tf.global_variables_initializer()
with tf.Session() as sess:         #开始训练
    sess.run(option)
    tf.train.start_queue_runners()
    print('开始训练')
    for i in range(epochs):
        images, labels = sess.run([train_images, train_labels])
        a, loss_value = sess.run([optimizer, loss], feed_dict={x: images, y: labels})
        print(f'Epoch:{i}, loss={loss_value}')
    #训练结束

    batches = int(math.ceil(test_num / batch_size))
    correct_num = 0
    total = batches * batch_size
    for i in range(batches):
        images, labels = sess.run([test_images, test_labels])
        pred = sess.run([top_k], feed_dict={x: images, y: labels})
        correct_num += np.sum(pred)
    print(f'Accuracy: {correct_num / total * 100}%')
