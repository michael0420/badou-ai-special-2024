import numpy as np
import tensorflow as tf
import os


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        初始化函数
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
        anchors_path: yolo anchor 文件路径
        classes_path: 数据集类别对应文件
        pre_train: 是否使用预训练darknet53模型
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    # 获取种类和先验框
    def _get_class(self):
        """
        获取类别名字
        返回: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        获取anchors
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # l2 正则化
    def _batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        对卷积层提取的feature map使用batch normalization
        input_layer: 输入的四维tensor
        name: batchnorm层的名字
        trainging: 是否为训练过程
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
        """
        bn_layer = tf.layers.batch_normalization(inputs=input_layer, momentum=norm_decay, epsilon=norm_epsilon,
                                                 center=True, scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    # 这个就是用来进行卷积的
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        """
        inputs: 输入变量
        filters_num: 卷积核数量
        strides: 卷积步长
        name: 卷积层名字
        trainging: 是否为训练过程
        use_bias: 是否使用偏置项
        kernel_size: 卷积核大小
        """
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filters_num,
            kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
            use_bias=use_bias, name=name)
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.99, norm_epsilon = 1e-3):
        """
        进行残差卷积的
        inputs: 输入变量
        filters_num: 卷积核数量
        trainging: 是否为训练过程
        blocks_num: block的数量
        conv_index: 为了方便加载预训练权重，统一命名序号
        weights_dict: 加载预训练模型的权重
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
        """
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        构建yolo3使用的darknet53网络结构
        inputs: 模型输入变量
        conv_index: 卷积层数序号，方便根据名字加载预训练权重
        weights_dict: 预训练权重
        training: 是否为训练
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
        """
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index),
                                                   training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index,  filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        inputs: 输入特征
        filters_num: 卷积核数量
        out_filters: 最后输出层的卷积核数量
        conv_index: 卷积层数序号，方便根据名字加载预训练权重
        training: 是否为训练
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
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
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1, name="conv2d_" + str(conv_index),
                                  use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    #
    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        返回三个特征层的内容
        inputs: 模型的输入变量
        num_anchors: 每个grid cell负责检测的anchor数量
        num_classes: 类别数量
        training: 是否为训练模式
        """
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training, norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)

            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]],
                                                          name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name='route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index=conv_index,
                                               training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]

