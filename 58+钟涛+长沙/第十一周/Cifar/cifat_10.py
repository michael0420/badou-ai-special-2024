import tensorflow as tf
import time
import CiFar10_data
import numpy as np
import math


max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"

#定义权重计算
def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(w1), w1, name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

#读取文件数据
images_train,lables_train = CiFar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,lables_test = CiFar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#定义输入占位符
x = tf.placeholder(tf.float32,[batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32,[batch_size])


#定义网络结构

#定义卷积
#第一层卷积
kernel1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=0.05, w1 = 0.0)
conv1 = tf.nn.conv2d(x,kernel1, strides=[1,1,1,1],padding = "SAME")
biase1 = tf.Variable(tf.constant(0.0,shape=[64]))
#y = ax +b
relue1 = tf.nn.relu(tf.nn.bias_add(conv1,biase1))
#池化
pool1 = tf.nn.max_pool(relue1, ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#第二层卷积
kernel2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=0.05, w1 = 0.0)
conv2 = tf.nn.conv2d(pool1,kernel2, strides=[1,1,1,1],padding = "SAME")
biase2 = tf.Variable(tf.constant(0.1,shape=[64]))
#y = ax +b
relue2 = tf.nn.relu(tf.nn.bias_add(conv2,biase2))
#池化
pool2 = tf.nn.max_pool(relue2, ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#转换为一维数组，用于FC; -1表示根据batch_size 自己计算维度
reshape = tf.reshape(pool2, [batch_size, -1])

#获取FC的输入大小
dim = reshape.get_shape()[1].value

#第一个全连接
w1 = variable_with_weight_loss(shape=[dim, 384],stddev=0.04, w1 =0.0004)
fc_bias1 = tf.Variable(tf.constant(0.1,shape=[384]))
fc_rele1 = tf.nn.relu(tf.matmul(reshape,w1) + fc_bias1)

#第二个全连接
w2 = variable_with_weight_loss(shape=[384, 192],stddev=0.04, w1 =0.0004)
fc_bias2 = tf.Variable(tf.constant(0.1,shape=[383]))
fc_rele2 = tf.nn.relu(tf.matmul(fc_rele1,w2) + fc_bias2)

#第三个全连接
w3 = variable_with_weight_loss(shape=[192,10],stddev=1/ 192.0, w1=0.0004)
fc_bias3 = tf.Variable(tf.constant(0.1,shape=[10]))
result = tf.nn.softmax(tf.matmul(fc_rele2,w3), fc_bias3)

#计算交叉熵
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, lables = tf.cast(y_,tf.int64))
#计算损失函数
weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

#优化器，反向传播
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#准确率排名
top_k_op = tf.nn.in_top_k(result, y_, 1)

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    #启动线程，文件下载使用了线程
    tf.train.start_queue_runners()

    # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        #下载数据
        image_batch,label_batch=sess.run([images_train,lables_train])
        _,loss_value = sess.run([train_op, loss],feed_dict={x:image_batch,y_:label_batch})
        duration = time.time() - start_time
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
            step, loss_value, examples_per_sec, sec_per_batch))


    #训练
    #math.ceil向上取整
    num = int(math.ceil(num_examples_for_eval/batch_size))
    true_count = 0
    for i in range(num):
        image_batch, label_batch = sess.run([images_test,lables_test])
        predictions=sess.run(top_k_op,feed_dict={x:image_batch, y_:label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / num_examples_for_eval) * 100))