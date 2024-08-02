import os
import numpy as np
import tensorflow as tf
import week15.yolo_homework.config_homework as config

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.classes = self._get_class()
        self.anchors = self._get_anchors()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    """
    设计网络结构
    """
    def _batch_normalization_layer(self, intput_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        标准化层
        :param intput_layer: 输入张量
        :param name: 命名
        :param training: 是否为训练
        :param norm_decay: 动量，用于计算移动平均值和移动方差
        :param norm_epsilon: 一个小数值，避免除以零
        :return: 激活后的结果
        """
        bn_layer = tf.layers.batch_normalization(inputs=intput_layer, name=name, momentum=norm_decay,
                                                 epsilon=norm_epsilon, center=True, scale=True,
                                                 training=training)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    def _conv2d_layer(self, input_layer, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
        卷积层
        :param input_layer:  输入张量
        :param filters_num:  卷积核数量
        :param kernel_size:  卷积核尺寸
        :param name:  名称
        :param use_bias:  是否使用偏置项
        :param strides:  步长
        :return:  特征提取出来的 feature map
        """
        conv = tf.layers.conv2d(inputs=input_layer, filters=filters_num, kernel_size=kernel_size, strides=[strides, strides],
                                kernel_initializer=tf.glorot_uniform_initializer(), padding=('SAME'if strides == 1 else 'VALID'),
                                name=name, use_bias=use_bias,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
        return conv

    def _Residual_Block(self, inputs, filters_num, block_num, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        残差网络
        :param inputs: 输入张量
        :param filters_num:  卷积核数量
        :param block_num:  block数量
        :param conv_index:  命名序号
        :param training:  是否为训练过程
        :param norm_decay: 动量，用于计算移动平均值和移动方差
        :param norm_epsilon: 一个小数值，避免除以零
        :return: layer： 残差网络结果
                 conv_index:  命名序号
        """
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [0, 1], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, name='conv2d_'+str(conv_index), strides=2)
        layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(block_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1, name='conv2d_'+str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name='conv2d_'+str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        dark53网络
        :param inputs: 输入张量
        :param conv_index:  命名序号
        :param training:  是否为训练过程
        :param norm_decay: 动量，用于计算移动平均值和移动方差
        :param norm_epsilon: 一个小数值，避免除以零
        :return:  route1: 返回第26层卷积计算结果52x52x256, 供后续使用
                  route2: 返回第43层卷积计算结果26x26x512, 供后续使用
                  conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
                  conv_index: 命名序号
        """
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1, name='conv2d_'+str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training,
                                                   norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_Block(conv, filters_num=64, block_num=1, training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon, conv_index=conv_index)
            conv, conv_index = self._Residual_Block(conv, filters_num=128, block_num=2, training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon, conv_index=conv_index)
            conv, conv_index = self._Residual_Block(conv, filters_num=256, block_num=8, training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon, conv_index=conv_index)
            route1 = conv

            conv, conv_index = self._Residual_Block(conv, filters_num=512, block_num=8, training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon, conv_index=conv_index)
            route2 = conv
            conv, conv_index = self._Residual_Block(conv, filters_num=1024, block_num=4, training=training, norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon, conv_index=conv_index)
        return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        针对3中比例的feature map的block，提高对于小物体的检测率
        :param inputs: 输入张量
        :param filters_num: 卷积核数量
        :param out_filters: 输出通道数
        :param conv_index:  命名序号
        :param training:  是否为训练过程
        :param norm_decay: 动量，用于计算移动平均值和移动方差
        :param norm_epsilon: 一个小数值，避免除以零
        :return:  route1: 返回第26层卷积计算结果52x52x256, 供后续使用
                  route2: 返回第43层卷积计算结果26x26x512, 供后续使用
                  conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
                  conv_index: 命名序号
        """
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training = training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1, name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        """
        返回三个特征层的结果，集成所有设计好的网络结构
        :param inputs:  输入张量
        :param num_anchors:  锚框数量
        :param num_classes:  类别数量
        :param training:  是否训练阶段
        :return:  返回三个特征层的结果
        """
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index=conv_index, training=training,
                                                                 norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.variable_scope("yolo"):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index,
                                                                training=training, norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)

            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index), training=training,
                                                      norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]],
                                                          name='upSample_0')
            route0 = tf.concat([upSample_0, conv2d_43], axis=-1, name='route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes * 5),
                                                                conv_index=conv_index, training=training, norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)

            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index), training=training,
                                                        norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]],
                                                          name='upSample_1')
            route1 = tf.concat([upSample_1, conv2d_26], axis=-1, name='route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index=conv_index,
                                               training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]

if __name__ == '__main__':
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, pre_train = False)
