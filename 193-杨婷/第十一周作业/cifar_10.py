import tensorflow as tf
import numpy as np
import time
import math
import cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_test = 10000
data_dir = 'cifar_data/cifar-10-batches-bin'


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss
def variable_with_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # 生成截断正态分布的随机数可以帮助模型在训练初期更稳定
    if w1 is not None:
        # L2正则化（即权重的平方和）乘以w1（越大表示对大的权重的惩罚越严厉，模型就越倾向于选择小的权重值）得到最终的惩罚分数
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('losses', weights_loss)  # 将weights_loss添加到名为"losses"的集合中，减少过拟合
    return var


# 使用cifar10_data里面已经定义好的文件序列读取函数,读取训练数据文件和测试数据文件.
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train, labels_train = cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层 shape=(kh,kw,ci,co)--> kernal_size, 输入通道大小(需要与上一个输入通道数一致)， 输出通道大小
kernal1 = variable_with_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # 5e-2-->0.05

conv1 = tf.nn.conv2d(x, kernal1, [1, 1, 1, 1], padding='SAME')
# [1, 1, 1, 1]表示batch, height, width, channels的步长
# 由于通常在batch和channels维度上不进行滑动（步长为1），所以通常设置为 [1, stride_height, stride_width, 1]

bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第二个卷积层
kernal2 = variable_with_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernal2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])  # -1代表的维度大小是除了batch_size之外的所有元素的乘积计算出来的。
dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后的第二个维度的值

# 建立第一个全连接层
weight1 = variable_with_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1)+fc_bias1)

# 建立第二个全连接层
weight2 = variable_with_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2)+fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

'''
计算损失，包括权重参数的正则化损失和交叉熵损失
tf.nn.sparse_softmax_cross_entropy_with_logits() 函数参数解释：
labels: 标签真实值，类型为 int32 或 int64 的张量，形状为 [batch_size]，包含每个样本的类别索引。注意，这些索引是“稀疏”的，意味着它们直接指向类别的索引，而不是 one-hot 编码。
logits: 模型原始输出，类型为 float32 或 float64 的张量，其形状为 [batch_size, num_classes]，包含了未经过 softmax 激活的原始预测值（也称为 logits）。
name (可选): 操作的名称。
dim (可选): 默认为 -1，指定 logits 的哪个维度对应于类别数。对于二维 logits，这通常是最后一个维度。
这个函数首先对每个样本的 logits 应用 softmax 函数，将其转换为概率分布。
然后，它计算每个样本的预测概率分布与其真实类别标签之间的交叉熵损失。
最终，cross_entropy 将是一个形状为 [batch_size] 的张量，包含了每个样本的交叉熵损失值
'''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))  # 将前面设置的集合里的元素相加，得到所有权重参数的正则化损失之和
# 求总loss
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss  # 对交叉熵损失做平均（因为result和y是批量处理的，cross_entropy包含多个样本的损失值）
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 学习率设置为 1e-3,minimize(loss)方法被用来告诉优化器要最小化的损失函数是loss

# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
# tf.nn.in_top_k 返回一个形状为 [batch_size]的布尔型张量，其中每个元素都表示对应样本的真实标签是否位于模型预测的前k个最高概率的类别中。
top_k_op = tf.nn.in_top_k(result, y, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])  # 运行，从训练数据集中获取一批图像和标签
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch})
        duration = time.time()-start_time

        if step % 100 == 0:
            examples_per_sec = batch_size/duration  # 计算每秒钟可以训练的样本数量,这是一个粗略的估计，因为它没有考虑到数据加载时间等外部因素
            sec_per_batch = float(duration)
            print('step %d, loss %.2f(%.1f examples/sec; %.3f sec/batch)'
                  % (step, loss_value, examples_per_sec, sec_per_batch))

    # 计算最终的正确率
    num_batch = int(math.ceil(num_examples_for_test/batch_size))  # math.ceil()函数用于向上求整,计算推理一共有多少个批次
    true_count = 0
    total_sample_count = min(num_batch*batch_size, num_examples_for_test)  # 计算总共推理了多少个样本

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print('accuracy = %.3f%%' % ((true_count/total_sample_count)*100))




