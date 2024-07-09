# _*_ coding: UTF-8 _*_
# @Time: 2024/7/1 20:32
# @Author: iris
# @Email: liuhw0225@126.com
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10Data


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


if __name__ == '__main__':
    max_steps = 4000
    batch_size = 100
    num_examples_for_eval = 10000
    data_dir = 'cifar_data/cifar-10-batches-bin'

    # 读取文件
    images_train, labels_train = Cifar10Data.inputs(data_dir, batch_size, distorted=True)
    images_test, labels_test = Cifar10Data.inputs(data_dir, batch_size, distorted=False)

    # 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
    x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    y_ = tf.placeholder(tf.int32, [batch_size])

    # 创建第一个卷积层  shape = (kh, kw, ci, co)
    kernel = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    bias = tf.Variable(tf.constant(0.0, shape=[64]))
    relu = tf.nn.relu6(tf.nn.bias_add(conv, bias))
    pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 创建第二个卷积层
    kernel = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
    conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding="SAME")
    bias = tf.Variable(tf.constant(0.1, shape=[64]))
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 因为要创建全链接层，所以使用tf.reshape()函数讲pool输出变为一维向量
    reshape = tf.reshape(pool, [batch_size, -1])
    # get_shape()[1].value表示获取reshape之后的第二个维度的值
    dim = reshape.get_shape()[1].value

    # 创建第一个全连接层
    weight = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
    fc_bias = tf.Variable(tf.constant(0.1, shape=[384]))
    fc = tf.nn.relu6(tf.add(tf.matmul(reshape, weight), fc_bias))

    # 建立第二个全连接层
    weight = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
    fc_bias = tf.Variable(tf.constant(0.1, shape=[192]))
    local = tf.nn.relu(tf.matmul(fc, weight) + fc_bias)

    # 建立第三个全连接层
    weight = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
    fc_bias = tf.Variable(tf.constant(0.1, shape=[10]))
    result = tf.add(tf.matmul(local, weight), fc_bias)

    # 计算损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))

    weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
    loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
    top_k_op = tf.nn.in_top_k(result, y_, 1)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
        tf.train.start_queue_runners()

        # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
        for step in range(max_steps):
            start_time = time.time()
            image_batch, label_batch = sess.run([images_train, labels_train])
            _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
            duration = time.time() - start_time

            if step % 100 == 0:
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
                    step, loss_value, examples_per_sec, sec_per_batch))

        # 计算最终的正确率
        num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
        true_count = 0
        total_sample_count = num_batch * batch_size

        # 在一个for循环里面统计所有预测正确的样例个数
        for j in range(num_batch):
            image_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
            true_count += np.sum(predictions)

        # 打印正确率信息
        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
