"""
定义神经网络模型
"""
import tensorflow.compat.v1 as tf
import tf_slim as slim

def VGG16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True):
    with tf.variable_scope(name_or_scope="vgg_16", default_name="vgg_16"):
        # 第1-2层：2 * conv + max_pool
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # 第3-4层：2 * conv + max_pool
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # 第5-7层：3 * conv + max_pool
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # 第8-10层：3 * conv + max_pool
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # 第11-13层：3 * conv + max_pool
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 第14层:利用卷积的模拟fc
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        # 第15层:利用卷积的模拟fc
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # 第16层:利用卷积的模拟fc
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

        return net