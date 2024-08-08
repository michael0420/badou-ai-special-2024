'''
yolo3的模型结构
'''

import numpy as np
import tensorflow as tf
import os


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        '''
        Introduction
        -----
        初始化函数
        Parameters
        -----
        :param norm_epsilon: 方差加上的极小数，防止除以0的情况
        :param norm_decay:在预测时，计算moving average时的衰减率
        :param anchors_path: yolo anchor的文件路径
        :param classes_path: 数据集类别对应的文件
        :param pre_train: 是否启用预训练的darknet53模型
        '''
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    ############################
    # 获取种类和先验框
    ############################
    def _get_class(self):
        '''
        Introduction
        ---------------
        获取类别名字
        :return:
            class_name:coco数据集类别对应的名字
        '''
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            classes_name = f.readlines()
        classes_name = [i.strip() for i in classes_name]
        return classes_name

    def _get_anchors(self):
        '''
        Introduction
        ------------
        获取先验框
        :return:

        '''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readlines()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    #########################
    # 用于生成层
    #########################
    # L2正则
    def _batch_normalization_layer(self, input_layer, name=None, training=True,
                                   norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Introduction:
        -------------
        对卷积层提取的feature map使用BN
        :param input_layer: 输入的思维tensor
        :param name: BN层名字
        :param training: 是否为训练过程
        :param norm_decay: 在与测试计算moving average时的衰减率
        :param norm_epsilon: 方差加上的极小数，防止除以0的情况
        :return:
            bn——layer: bantch normalization处理之后的feature map
        '''
        bn_layer = tf.layers.batch_normalization(inputs=input_layer, momentum=norm_decay, epsilon=norm_epsilon,
                                                 center=True, scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    # 用于卷积
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        '''
        使用tf.layers.conv2d减少权重和偏执矩阵初始化过程，以及卷积后加上偏执项的操作
        经过卷积之后需要进行batch norm。最后使用leaky Relu激活函数
        根据卷积的步长，如果步长为2，则对图像进行降采样
        如：输入（416*416），卷积核为3.stride=2时，（416-3+2）/2+1=208.相当于做了池化，
        stride>1 ,做padding操作，采用'same'方式
        :param inputs: 输入变量
        :param filters_num: 卷积核数量
        :param kernel_size: 卷积核大小
        :param name: 卷积层名字
        :param use_bias: 是否使用偏置项
        :param stride: 步长
        :return:
            conv：卷积后的feature map
        '''
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filters_num, kernel_size=kernel_size, strides=[strides, strides],
            kernel_initializer=tf.glorot_uniform_initializer(), padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name
        )
        return conv

    # 残差卷积
    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True,
                        norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Darknet的残差block，类似resnet的两层卷积结构，分别采用1*1和3*3的卷积核，使用1*1卷积核是为了减少channel维度
        :param inputs: 输入变量
        :param filters_num: 卷积核数量
        :param blocks_num: block数量
        :param conv_index: 为了方便加载预训练权重，统一命名序号
        :param training: 是否为训练过程
        :param norm_decay: 在预测时，计算moving average的衰减率
        :param norm_epsilon:方差加上的极小值，防止除以0的情况
        weights_dict: 加载预训练模型的权重
        :return:
            inputs：经过残差网络处理后的结果
        '''
        # 对输入的feature map进行padding
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name='conv2d_'+str(conv_index))
        layer = self._batch_normalization_layer(
            layer, name='batch_normalization_'+str(conv_index), training=training,
            norm_decay=norm_decay, norm_epsilon=norm_epsilon
        )
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(
                layer, filters_num // 2, kernel_size=1, strides=1, name='conv2d_'+str(conv_index)
            )
            layer = self._batch_normalization_layer(
                layer, name='batch_normalization_'+str(conv_index), training=training,
                norm_decay=norm_decay, norm_epsilon=norm_epsilon
            )
            conv_index += 1
            layer = self._conv2d_layer(
                layer, filters_num, kernel_size=3, strides=1, name='conv2d_' + str(conv_index)
            )
            layer = self._batch_normalization_layer(
                layer, name='batch_normalization_' + str(conv_index), training=training,
                norm_decay=norm_decay, norm_epsilon=norm_epsilon
            )
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    ###############################
    # 生成darknet53
    ###############################
    def darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Introduction:
        ----------------------
        构建yolo3使用的darknet53网络结构
        :param inputs: 输入变量
        :param conv_index: 卷积序号
        weights_dict: 预训练权重
        :param training: 是否为训练
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param norm_epsilon:方差上加的极小值，防止除以0的情况
        :return:
            conv：经过52层卷积后的结果，输入图片416*416*3，输出shape：13*13*1024
            route1：返回第26层卷积结果，52*52*256
            route2：返回第43层卷积结果，26*26*512
            conv_index:卷积层计数
        '''
        with tf.variable_scope('darknet53'):
            # 416*416*3>>>516*416*32
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index),
                                                   training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1

            # 416*416*32>>>208*208*64
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 208*208*64>>104*104*128
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 104*104*128>>>52*52*256
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            # 52*52*256>>>26*26*512
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            # 26*26*512>>>13*13*1024
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        '''
        yolo3在darknet53提取的特征层基础上，又加了争对3种不同比例的feature map的block，提高对小物体的检测率
        # 输出两个网络结果
        # 第一个是进行5次卷积后，用于下一次逆卷积的，卷积过程是1X1，3X3，1X1，3X3，1X1
        # 第二个是进行5+2次卷积，作为一个特征层的，卷积过程是1X1，3X3，1X1，3X3，1X1，3X3，1X1

        :param inputs: 输入特征
        :param filters_num: 卷积核数量
        :param out_filters: 最后输出层的卷积核数量
        :param conv_index: 卷积层数序号
        :param training: 是否在训练
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param norm_epsilon: 方差加上的极小值，防止出现除以0的情况
        :return:
            route:返回最后一层卷积的前一层结果
            conv：返回最后一层卷积的结果
            conv_index:conv层数序号
        '''
        #1*1
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1, name='conv2d_'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        #3*3
        conv = self._conv2d_layer(inputs, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        # 1*1
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        # 3*3
        conv = self._conv2d_layer(inputs, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        # 1*1
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv

        # 3*3
        conv = self._conv2d_layer(inputs, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        # 1*1
        conv = self._conv2d_layer(inputs, filters_num=out_filters, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index),use_bias=True)
        conv_index += 1

        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        '''
        返回三个特征层的内容
        构建yolo模型
        :param inputs: 输入的变量
        :param num_anchors: 每个grid cell负责检查的anchor数量
        :param num_classes: 类别数量
        :param training: 是否为训练模式
        :return:
        '''
        conv_index = 1
        # route1=52*52*256, route2=26*26*512, route3=13*13*1024
        conv2d_26, conv2d_43, conv, conv_index = self.darknet53(inputs, conv_index, training=training,
                                                                norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        with tf.variable_scope('yolo'):
            ############
            # 获得第一个特征层
            ############
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors*(num_classes+5), conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            ############
            # 获得第二个特征层
            ############
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # unSample_0=26*26*256
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2*tf.shape(conv2d_60)[1], 2*tf.shape(conv2d_60)[2]], name='upSample_0')
            # route0= 26*26*768
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name='route_0')
            # conv2d_65 = 26,26,256，conv2d_67 = 26,26,255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            ############
            # 获得第三个特征层
            ############
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # unSample_1=52*52*128
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,
                                                          [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[2]],
                                                          name='upSample_1')
            # route1= 52,52,384
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')
            # conv2d_75=52,52,
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
        return [conv2d_59, conv2d_67, conv2d_75]