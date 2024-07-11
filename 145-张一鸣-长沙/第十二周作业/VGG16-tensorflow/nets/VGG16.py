# coding= utf-8

'''
    用 tensorflow 实现 VGG16 模型
'''

import tensorflow as tf


# 创建slim对象
slim = tf.contrib.slim

def VGG_16(input, classes=1000, is_training=True, dropout=0.5, spatial_squeeze=True, scope='vgg_16'):
    # VGG16网络构建
    # input(224, 224, 3)
    # conv2D  (3, 3) * 2 输出(224, 224, 64)
    # maxpool (2, 2)     输出(112, 112, 64)
    # conv2D  (3, 3) * 2 输出(112, 112, 128)
    # maxpool (2, 2)     输出(56, 56, 128)
    # conv2D  (3, 3) * 3 (56, 56, 256)
    # maxpool (2, 2)     输出 (28, 28, 256)
    # conv2D  (3, 3) * 3 (28, 28, 512)
    # maxpool (2, 2)     输出 (14, 14, 512)
    # conv2D  (3, 3) * 3 (14, 14, 512)
    # maxpool (2, 2)     输出 (7, 7, 512)
    # fc (7, 7, 4096)(1, 1, 4096)
    # fc (1, 1, 1000)
    # softmax
    with tf.variable_scope(scope, 'vgg_16', [input]):

        vgg16_net = slim.repeat(input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        vgg16_net = slim.max_pool2d(vgg16_net, [2, 2], scope='maxpool1')

        vgg16_net = slim.repeat(vgg16_net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        vgg16_net = slim.max_pool2d(vgg16_net, [2, 2], scope='maxpool2')

        vgg16_net = slim.repeat(vgg16_net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        vgg16_net = slim.max_pool2d(vgg16_net, [2, 2], scope='maxpool3')

        vgg16_net = slim.repeat(vgg16_net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        vgg16_net = slim.max_pool2d(vgg16_net, [2, 2], scope='maxpool4')

        vgg16_net = slim.repeat(vgg16_net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        vgg16_net = slim.max_pool2d(vgg16_net, [2, 2], scope='maxpool5')

        # 魔改利用卷积进行全连接
        vgg16_net = slim.conv2d(vgg16_net, 4096, [7, 7], padding='valid', scope='fc6')
        vgg16_net = dropout(vgg16_net, dropout, is_training=is_training, scope='dropout6')

        vgg16_net = slim.conv2d(vgg16_net, 4096, [1, 1], padding='valid', scope='fc7')
        vgg16_net = dropout(vgg16_net, dropout, is_training=is_training, scope='dropout7')

        vgg16_net = slim.conv2d(vgg16_net, classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            vgg16_net = tf.squeeze(vgg16_net, [1, 2], name='squeezed')

        return vgg16_net

