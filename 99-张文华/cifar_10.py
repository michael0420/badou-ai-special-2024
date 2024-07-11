'''
使用tf构建卷积神经网络的整体结构，并实现对cifar_10数据的训练
'''
import tensorflow as tf
import numpy as np
import time
import math
import Cifar_data

data_dir = 'cifar_data/cifar-10-batches-bin'
max_steps = 4000
batch_size =100
num_examples_for_eval = 10000

# 读取训练数据和测试数据
img_train, label_train = Cifar_data.inputs(data_dir, batch_size, distorted=True)
img_test, label_test = Cifar_data.inputs(data_dir, batch_size, distorted=False)

# 创建x,y两个placeholder，用于在训练和测试时提供输入数据及标签
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])


# 创建一个variable_with_weight_loss()函数，
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


# 创建第一个卷积层 shape=(kh,kw,ci,co)
kernel_1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel_1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第二个卷积层
kernel_2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel_2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 进如FC，数据需转为一维
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
# get_shape()[1].value表示获取reshape之后的第二个维度的值

# 创建第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_relu1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 创建第二个连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_relu2 = tf.nn.relu(tf.matmul(fc_relu1, weight2) + fc_bias2)

# 创建第三个链接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(fc_relu2, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=result,labels=tf.cast(y_, tf.int64)
)
weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，
# 即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y_, 1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作
    tf.train.start_queue_runners()
    # 每隔100step会计算并展示当前的loss、
    # 每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        img_batch, label_batch = sess.run([img_train, label_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x:img_batch, y_:label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_bantch = float(duration)
            print(f'step{step}, loss={loss_value:.2f}'
                  f'({examples_per_sec:.2f}examples/sec;{sec_per_bantch:.2f}sec/batch)')

    # 计算正确率
    num_batch = int(math.ceil(num_examples_for_eval/batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环中统计所有预测的正确率
    for j in range(num_batch):
        img_batch, label_batch = sess.run([img_test, label_test])
        predictions = sess.run([top_k_op], feed_dict={x:img_batch,y_:label_batch})
        true_count += np.sum(predictions)

    # 打印正确率
    print(f'预测的准确率：{true_count / total_sample_count * 100:.2f}')
