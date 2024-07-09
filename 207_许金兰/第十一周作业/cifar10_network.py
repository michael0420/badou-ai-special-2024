import tensorflow as tf
import numpy as np
import math
from Cifar10_data import Cifar10DataProcess


def weight_with_loss(shape, stddev, w1):
    """
    使用参数w1控制L2 loss的大小
    :param shape:参数矩阵的形状
    :param stddev:标准差
    :param w1:权重
    :return:
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


# 初始化参数
max_steps = 1000
batch_size = 32
num_examples_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"
# 读取数据
cifar10 = Cifar10DataProcess(data_dir)
train_imgs, train_labels = cifar10.data_process(batch_size, enhance=True)
test_imgs, test_labels = cifar10.data_process(batch_size, enhance=False)
# 设置两个变量
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 第一个卷积层
kernel1 = weight_with_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第二个卷积层
kernel2 = weight_with_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 将数据拉平成一维
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 第一个全连接层
weight1 = weight_with_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 第二个全连接层
weight2 = weight_with_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 第三个全连接层
weight3 = weight_with_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_operation = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 计算topk准确率
top_k = tf.nn.in_top_k(result, y_, 1)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 启动多线程操作
    tf.train.start_queue_runners()

    for step in range(max_steps):
        # 获取数据
        image_batch, label_batch = sess.run([train_imgs, train_labels])
        # 模型训练
        _, loss_value = sess.run([train_operation, loss], feed_dict={x: image_batch, y_: label_batch})
        # 每隔100step会计算并展示当前的loss
        if step % 100 == 0:
            print("step %d,loss=%.2f" % (step, loss_value))

    # 计算最终的正确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size
    # 统计正确率
    for j in range(num_batch):
        # 获取测试数据
        image_batch, label_batch = sess.run([test_imgs, test_labels])
        # 模型预测
        predictions = sess.run([top_k], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
