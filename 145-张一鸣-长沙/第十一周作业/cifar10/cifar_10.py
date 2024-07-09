# coding = utf-8

'''
    用tensorflow实现卷积神经网络cifar-10分类
'''


import numpy as np
import time
import math
import Cifar10_data
import tensorflow as tf
tf.disable_v2_behavior()


max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = 'Cifar_data/cifar-10-batches-bin'


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('losses', weights_loss)
    return var

#  使用Cifar10_data里面已经定义好文件序列读取函数/训练数据文件/测试数据文件.
#  其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
img_train, labels_train = Cifar10_data.inputs(data_dir=data_dir,
                                              batch_size=batch_size,
                                              distorted=True)
img_test, labels_test = Cifar10_data.inputs(data_dir=data_dir,
                                            batch_size=batch_size,
                                            distorted=None)

#  创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
#  要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层 shape=(kh,kw,ci,co)
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第二个卷积层 shape=(kh,kw,ci,co)
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 因为要进行全连接层的操作
# 所以这里使用tf.reshape()函数将pool2输出变成一维向量,-1代表将pool2的三维结构拉直为一维结构
reshape = tf.reshape(pool2, [batch_size, -1])
# 并使用get_shape()函数获取扁平化之后的长度
# get_shape()[1].value表示获取reshape之后的第二个维度的值
dim = reshape.get_shape()[1].value

# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_b1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_b1)

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_b2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1, weight2) + fc_b2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_b3 = tf.Variable(tf.constant(0.1, shape=[10]))
fc3 = tf.add(tf.matmul(fc2, weight3), fc_b3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3,
                                                               labels=tf.cast(y, tf.int64))
weight_with_12_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weight_with_12_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率
# 函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(fc3, y, 1)

init_op = tf.global_variables_initializer()

# Session.run()
with tf.Session() as ss:
    ss.run(init_op)
    # 启动线程操作，因为之前数据增强时使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()
    # 每隔100 step计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        img_batch, label_batch = ss.run([img_train, labels_train])
        _, loss_value = ss.run([train_op, loss], feed_dict={x: img_batch, y: label_batch})
        time_cost = time.time() - start_time

        if step % 100 == 0:
            num_examples_per_sec = batch_size / time_cost
            sec_per_batch = float(time_cost)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step, loss_value, num_examples_per_sec, sec_per_batch))

    # 计算正确率,math.ceil()函数用于求整
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    sum_correct = 0
    all_data = num_batch * batch_size

    # 在一个for循环里面统计所有推理正确的样例个数
    for i in range(num_batch):
        img_batch, label_batch = ss.run([img_test, labels_test])
        predict = ss.run([top_k_op], feed_dict={x: img_batch, y: label_batch})
        sum_correct += np.sum(predict)

    print('推理准确率：%.4ff%%' % ((sum_correct / all_data) * 100))
