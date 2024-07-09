import tensorflow as tf
import readcifar10

slim = tf.contrib.slim
import os
import resnet


# 定义一个模型
def model(image, keep_prob=0.8, is_training=True):
    batch_norm_params = {
        # 对batch_norm层添加约束 train:true,test:false
        "is_training": is_training,
        # 防止batch_norm归一化的时候除0
        "epsilon": 1e-5,
        # 衰减系数
        "decay": 0.997,
        'scale': True,
        # 对batch_norm参数收集
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    # 定义优化器#
    with slim.arg_scope(
            [slim.conv2d],
            # 采用方差尺度不变的方式进行权值初始化
            weights_initializer=slim.variance_scaling_initializer(),
            # 定义卷积层激活函数：relu
            activation_fn=tf.nn.relu,
            # 正则化约束方法：L2正则 权值0.0001
            weights_regularizer=slim.l2_regularizer(0.0001),
            # 定义batch_norm
            normalizer_fn=slim.batch_norm,
            # 指定batch_norm参数
            normalizer_params=batch_norm_params):
        # 保证每次池化之后，尺寸是等比例变化的 #
        with slim.arg_scope([slim.max_pool2d], padding="SAME"):
            # 卷积1 输入参数为image，卷积核3*3
            net = slim.conv2d(image, 32, [3, 3], scope='conv1')
            # 卷积2 输入参数为conv2的输出
            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            # 池化层，进行两倍下采样
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.conv2d(net, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv5')
            net = slim.conv2d(net, 128, [3, 3], scope='conv6')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv7')
            # nhwc--n11c 通过reduce_mean实现 average pooling 对当前特征图求均值
            net = tf.reduce_mean(net, axis=[1, 2])
            net = slim.flatten(net)
            # 全连接层
            net = slim.fully_connected(net, 1024)
            # dropout层 对神经元进行正则化，keep_prob：训练时<1 测试时=1
            slim.dropout(net, keep_prob)
            net = slim.fully_connected(net, 10)
        # 10 dim vec 10维向量
        return net


# 定义loss 交叉熵损失函数,输入(logits预测出的概率分布值,label),输出 loss

def loss(logits, label):
    # 对lable进行onehot编码，定义onehot长度为10
    one_hot_label = slim.one_hot_encoding(label, 10)
    # 交叉熵损失，传入预测结果和label
    slim.losses.softmax_cross_entropy(logits, one_hot_label)
    # 获取正则化loss集合
    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # 将l2_loss相加计算出整体的l2_loss
    l2_loss = tf.add_n(reg_set)
    # 将l2_loss添加到loss中
    slim.losses.add_loss(l2_loss)
    # 计算出所有的loss
    totalloss = slim.losses.get_total_loss()

    return totalloss, l2_loss


"""定义优化器"""


def func_optimal(batchsize, loss_val):
    global_step = tf.Variable(0, trainable=False)
    # 定义学习率的变化
    lr = tf.train.exponential_decay(0.01,  # 初始学习率
                                    global_step,
                                    decay_steps=50000 // batchsize,  # 每次衰减对应的步长=训练样本数/batchsize
                                    decay_rate=0.95,  # 每次衰减0.95
                                    staircase=False)  # True：阶梯式下降，False平滑下降
    # 收集和batch_norm相关参数
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 对batch_norm层进行更新
    with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step)
    return global_step, op, lr


"""搭建训练代码"""


def train():
    # 定义batchsize
    batchsize = 64
    # 定义日志存放目录
    floder_log = 'logdirs-resnet'
    # 定义model存放路径
    floder_model = 'model-resnet'

    if not os.path.exists(floder_log):
        os.mkdir(floder_log)

    if not os.path.exists(floder_model):
        os.mkdir(floder_model)

    # 定义训练样本日志
    tr_summary = set()
    # 定义测试样本日志
    te_summary = set()

    ##data
    tr_im, tr_label = readcifar10.read(batchsize, 0, 1)
    te_im, te_label = readcifar10.read(batchsize, 1, 0)

    ##net
    # 定义输入数据
    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3],
                                name='input_data')

    input_label = tf.placeholder(tf.int64, shape=[None],
                                 name='input_label')
    keep_prob = tf.placeholder(tf.float32, shape=None,
                               name='keep_prob')

    is_training = tf.placeholder(tf.bool, shape=None,
                                 name='is_training')
    logits = resnet.model_resnet(input_data, keep_prob=keep_prob, is_training=is_training)

    ##loss

    total_loss, l2_loss = loss(logits, input_label)

    # 记录loss的日志信息
    tr_summary.add(tf.summary.scalar('train total loss', total_loss))
    tr_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    te_summary.add(tf.summary.scalar('train total loss', total_loss))
    te_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    ##accurancy
    # 获取当前概率分布中最大的值所对应的索引
    pred_max = tf.argmax(logits, 1)
    # 判断这个值是否和label相等
    correct = tf.equal(pred_max, input_label)
    # 定义精度
    accurancy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # 记录精度的日志信息
    tr_summary.add(tf.summary.scalar('train accurancy', accurancy))
    te_summary.add(tf.summary.scalar('test accurancy', accurancy))

    ##op 调用优化器
    global_step, op, lr = func_optimal(batchsize, total_loss)
    # 记录学习率的日志信息
    tr_summary.add(tf.summary.scalar('train lr', lr))
    te_summary.add(tf.summary.scalar('test lr', lr))

    tr_summary.add(tf.summary.image('train image', input_data * 128 + 128))
    te_summary.add(tf.summary.image('test image', input_data * 128 + 128))

    with tf.Session() as sess:
        # 全局变量、局部变量初始化
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        # 启动文件队列写入线程
        tf.train.start_queue_runners(sess=sess,
                                     coord=tf.train.Coordinator())  # 启动多线程管理器

        # 模型存储，最大5个
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        # 获取模型文件中最新的model
        ckpt = tf.train.latest_checkpoint(floder_model)

        if ckpt:
            saver.restore(sess, ckpt)

        epoch_val = 10

        # 通过summary.merge合并日志信息
        tr_summary_op = tf.summary.merge(list(tr_summary))
        te_summary_op = tf.summary.merge(list(te_summary))

        summary_writer = tf.summary.FileWriter(floder_log, sess.graph)

        for i in range(50000 * epoch_val):
            # 获取训练样本
            train_im_batch, train_label_batch = \
                sess.run([tr_im, tr_label])
            # 定义feed
            feed_dict = {
                input_data: train_im_batch,
                input_label: train_label_batch,
                keep_prob: 0.8,
                is_training: True
            }

            """完成对参数的更新"""
            _, global_step_val, \
            lr_val, \
            total_loss_val, \
            accurancy_val, tr_summary_str = sess.run([op,
                                                      global_step,
                                                      lr,
                                                      total_loss,
                                                      accurancy, tr_summary_op],
                                                     feed_dict=feed_dict)
            # 将后端运行结果写入日志
            summary_writer.add_summary(tr_summary_str, global_step_val)

            # 每100次打印一次训练集
            if i % 100 == 0:
                print("{},{},{},{}".format(global_step_val,
                                           lr_val, total_loss_val,
                                           accurancy_val))

            if i % (50000 // batchsize) == 0:
                test_loss = 0
                test_acc = 0
                for ii in range(10000 // batchsize):
                    test_im_batch, test_label_batch = \
                        sess.run([te_im, te_label])
                    feed_dict = {
                        input_data: test_im_batch,
                        input_label: test_label_batch,
                        keep_prob: 1.0,
                        is_training: False
                    }

                    total_loss_val, global_step_val, \
                    accurancy_val, te_summary_str = sess.run([total_loss, global_step,
                                                              accurancy, te_summary_op],
                                                             feed_dict=feed_dict)

                    # 将测试运行结果写入日志
                    summary_writer.add_summary(te_summary_str, global_step_val)

                    test_loss += total_loss_val
                    test_acc += accurancy_val
                # 输出测试集平均精度和loss
                print('test：', test_loss * batchsize / 10000,
                      test_acc * batchsize / 10000)

            # 每隔1000次保存一次模型
            if i % 1000 == 0:
                saver.save(sess, "{}/model.ckpt{}".format(floder_model, str(global_step_val)))
    return


# main 运行脚本
if __name__ == '__main__':
    train()
