# _*_ coding: UTF-8 _*_
# @Time: 2024/7/8 21:31
# @Author: iris
# @Email: liuhw0225@126.com
import tensorflow as tf

# 创建slim对象
slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    """
    vgg 16 网络结构
    :param inputs: 输入信息
    :param num_classes: 分类个数
    :param is_training: 是否训练
    :param dropout_keep_prob: 防止过拟合
    :param spatial_squeeze: 特殊设置
    :param scope: 名称
    :return:
    """

    """
        with 关键字是一个替你管理实现上下文协议对象的东西，适用于对资源进行访问的场合，
        确保不管使用过程中是否发生异常都会执行必要的"清楚"操作，释放资源
        比如文件使用后自动关闭、线程中锁的自动获取和释放等.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        """
        # 建立网络 
            conv - 2
            
            pool - 1
            conv - 2
            
            pool - 1
            conv - 3
            
            pool - 1
            conv - 3
            
            pool - 1
            conv - 3
            
            pool - 1
            fc   - 3
            softmax - 1
        """
        # 重复2次，使用[3,3]卷积核, 输出64层 -- (224, 224, 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # [2,2]最大池化, 输出(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # 重复2次，使用[3,3]卷积核, 输出128层 -- (112, 112, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # [2,2]最大池化, 输出(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # 重复3次，使用[3,3]卷积核, 输出256层 -- (56, 56, 256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # [2,2]最大池化, 输出(28, 28, 256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # 重复3次，使用[3,3]卷积核, 输出512层 -- (28, 28, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # [2,2]最大池化, 输出(14, 14, 512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # 重复3次，使用[3,3]卷积核, 输出256层 -- (14, 14, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # [2,2]最大池化, 输出(7, 7, 512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积方式实现全FC, 输出位(1, 1, 4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # 输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
