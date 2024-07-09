import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data


max_steps = 4000
batch_size = 100
num_example_for_eval = 10000
data_dir = './cifar_data/cifar-10-batches-bin'

def varuable_with_weight_loss(shape,stddev,w1):
    #生成截断的正态分布张量
    #这意味着生成的数值大致遵循正态分布（高斯分布），但是极端的大值和小值会被截断
    #避免生成过于极端的异常值，有助于神经网络的稳定训练
    #它可以帮助网络权重在训练初期处于合理的范围内，
    # 避免梯度消失或梯度爆炸问题，从而使网络更容易收敛。
    # 在神经网络中，权重的初始化对于训练过程的稳定性有着重要影响，
    # 因此选择合适的初始化方法是很重要
    var = tf.Variable(tf.truncated_normal(shape,stddev))
    if w1 is not None:
        #loss得到的值
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loass')
        tf.add_to_collection('losses',weight_loss)
    return var


image_train, label_train = Cifar10_data.inputs(data_dir,batch_size,distorted=True)
image_test, label_test = Cifar10_data.inputs(data_dir,batch_size,distorted=False)

x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y = tf.placeholder(tf.int32,[batch_size])
# print(image_train)
# print("---")
# print(label_train)

# #创建卷积层
kernel1=varuable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

# #
kernel2=varuable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#
#
# #
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value
#
# #建立第一个全连接层
weight1=varuable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

##建立第二个全连接层
weight2=varuable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)
#
# #建立第三个全连接层
weight3=varuable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))
#
weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
#
top_k_op=tf.nn.in_top_k(result,y,1)
#
init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([image_train,label_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))
    num_batch = int(math.ceil(num_example_for_eval / batch_size))  # math.ceil()函数用于求整
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, label_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))














