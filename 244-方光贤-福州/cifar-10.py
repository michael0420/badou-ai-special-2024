import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"

# 创建一个variable_with_weight_loss()函数，该函数的作用是：
# 1.使用参数w1控制L2 loss的大小
# 2.使用函数tf.nn.l2_loss()计算权重L2 loss
# 3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
# 4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var

#其中训练数据文件进行数据增强处理 测试数据文件不进行数据增强处理
images_train, labels_train = Cifar10_data.inputs(data_dir = data_dir, batch_size = batch_size,distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建x和y用于在训练或评估时提供输入的数据和标签
# 第一个参数是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层 shape=(kh,kw,ci,co)
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 为了全连接 这里使用tf.reshape()函数将pool2输出变成一维向量 使用get_shape()函数获取长度
reshape = tf.reshape(pool2, [batch_size, -1])  # 参数-1将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后的第二个维度的值

# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1)+fc_bias1)

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

# 计算损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))

weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy)+weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数tf.nn.in_top_k()计算输出结果中top k的准确率 取1代表最高概率
top_k_op = tf.nn.in_top_k(result, y_, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()      

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (step, loss_value, examples_per_sec, sec_per_batch))

# 计算准确率
    num_batch = int(math.ceil(num_examples_for_eval/batch_size))  # math.ceil()向上取整
    true_count=0
    total_sample_count=num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch=sess.run([images_test,labels_test])
        predictions = sess.run([top_k_op],feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))
