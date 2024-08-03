import numpy as np
import tensorflow as tf
import os

class yolo3:
    def __init__(self, norm_epsilon, norm_decay, archors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.archors_path = archors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._getanchors()
        self.classes = self._get_class()

    def _get_class(self):
        with open(self.classes_path) as f:
            class_name = f.readlines()
        class_names = [c.strip() for c in class_name]
        return class_names

    def _getanchors(self):
        with open(self.archors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #将bn和relu放在一个方法中
    #当center=True时，
    # Batch Normalization层会学习一个偏置向量，
    # 该向量会加到标准化后的输出上，
    # 使得输出的均值不一定是0。
    # 这相当于在标准化输出的基础上加上一个可学习的偏置项
    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        bn_layer = tf.layers.batch_normalization(inputs=input_layer, momentum=norm_decay, epsilon=norm_epsilon,
                                                 center=True, scale=True, training=training, name=name)
        #在深层网络中leaky_relu用来替换relu存在的梯度消失
        bn_layer_leaky_relu = tf.nn.leaky_relu(bn_layer, alpha=0.1)
        return bn_layer_leaky_relu

    def _conv2d_layer(self, inputs, filters_nums, kernel_size, name, strides=1, use_bias=False):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters_nums, kernel_size=kernel_size,
                                strides=[strides, strides],
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                                padding=('SAME' if strides == 1 else 'VALID'), use_bias=use_bias, name=name)
        return conv

    def _Residual_block(self, inputs, filters_nums, block_nums, conv_index, training=True, norm_decay=0.99,
                        norm_epsilon=1e-3):
        #由于要做strides=2的卷积，且希望输出长度不变，batchsize和通道数不做填充,在长和宽位置做一圈填充
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_nums=filters_nums, kernel_size=3, strides=2,
                                   name='conv_2d_' + str(conv_index))
        layer = self._batch_normalization_layer(layer, training=training, norm_decay=norm_decay,
                                                norm_epsilon=norm_epsilon, name='bn_' + str(conv_index))
        conv_index += 1
        for _ in range(block_nums):
            shortcut = layer
            #做1*1卷积压缩特征
            layer = self._conv2d_layer(layer, filters_nums=filters_nums // 2, kernel_size=1, strides=1,
                                       name='conv_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name='bn_' + str(conv_index), training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            #再做一次3*3但是步长为1的卷积
            layer = self._conv2d_layer(layer, filters_nums, kernel_size=3, strides=1, name='conv_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name='bn_' + str(conv_index), training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    #生成最终的darknet53结构网络
    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        with tf.variable_scope('darknet53'):
            #416,416,3 - 416,416,32
            conv = self._conv2d_layer(inputs, filters_nums=32, kernel_size=3, strides=1,
                                      name='conv_1_darknet_' + str(conv_index))
            conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                                   norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1

            #416,416,32  - 208,208,64
            conv, conv_index = self._Residual_block(inputs=conv, filters_nums=64, block_nums=1, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            #208,208,64 -104,104,128
            conv, conv_index = self._Residual_block(inputs=conv, filters_nums=128, block_nums=2, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            #104,104,128 - 52,52, 256
            conv, conv_index = self._Residual_block(inputs=conv, filters_nums=256, block_nums=8, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            #52,52, 256 - 26,26,512
            conv, conv_index = self._Residual_block(inputs=conv, filters_nums=512, block_nums=8, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            #26,26,512 - 13,13,1024
            conv, conv_index = self._Residual_block(inputs=conv, filters_nums=1024, block_nums=4, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

            return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filter_num, out_filters, conv_index, training=True, norm_decay=0.99,
                    norm_epsilon=1e-3):
        #第一次1*1
        conv = self._conv2d_layer(inputs=inputs, filters_nums=filter_num, kernel_size=1, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        #第二次3*3
        conv = self._conv2d_layer(inputs=conv, filters_nums=filter_num * 2, kernel_size=3, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        #第三次1*1
        conv = self._conv2d_layer(inputs=conv, filters_nums=filter_num, kernel_size=1, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        #第四次3*3
        conv = self._conv2d_layer(inputs=conv, filters_nums=filter_num * 2, kernel_size=3, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        #第五次1*1
        conv = self._conv2d_layer(inputs=conv, filters_nums=filter_num, kernel_size=1, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        #输出结果的特征还要做一个3*3和1*1，得到最终结果之一
        conv = self._conv2d_layer(inputs=conv, filters_nums=filter_num * 2, kernel_size=3, strides=1,
                                  name='conv_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='bn_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(inputs=conv, filters_nums=out_filters, kernel_size=1, strides=1,
                                  name='conv_' + str(conv_index))
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        conv_index = 1
        conv26, conv43, conv, conv_index = self._darknet53(inputs, conv_index, training=training,
                                                           norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
            #第一个特征输出,conv59已经是了
            conv57, conv59, conv_index = self._yolo_block(inputs=conv, filter_num=512,
                                                          out_filters=num_anchors * (num_classes + 5),
                                                          conv_index=conv_index, training=training)
            #第二个特征层
            #首先对conv57做卷积和上采样
            conv60 = self._conv2d_layer(inputs=conv57, filters_nums=256, kernel_size=1, strides=1,
                                        name='conv_' + str(conv_index))
            conv60 = self._batch_normalization_layer(conv60, name='bn_' + str(conv_index), training=training)
            conv_index += 1
            #上采样
            unSample_0 = tf.image.resize_nearest_neighbor(conv60, [tf.shape(conv60)[1] * 2, tf.shape(conv60)[1] * 2],
                                                          name='unSample_0')
            #上采样结果和conv43做合并)
            route0 = tf.concat([unSample_0, conv43], axis=-1, name='route_0')

            #conv67已经是第二个特征层的输出了
            conv65, conv67, conv_index = self._yolo_block(inputs=route0, filter_num=256,
                                                          out_filters=num_anchors * (num_classes + 5),
                                                          conv_index=conv_index, training=training)
            #第三个特征层
            conv68 = self._conv2d_layer(inputs=conv65, filters_nums=128, kernel_size=1, strides=1,
                                        name='conv_' + str(conv_index))
            conv68 = self._batch_normalization_layer(conv68, name='bn_' + str(conv_index), training=training,
                                                     norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv68, [tf.shape(conv68)[1] * 2, tf.shape(conv68)[1] * 2],
                                                          name='unSample_1')
            #合并
            route_1 = tf.concat([conv26, unSample_1], axis=-1, name='route_1')
            _, conv75, _ = self._yolo_block(route_1, 128, num_anchors * (num_classes + 5), conv_index=conv_index,
                                            training=training)

            return [conv59, conv67, conv75]


if __name__ == '__main__':
    input_shape = (1, 416, 416, 3)
    inputs = tf.placeholder(dtype=tf.float32, shape=input_shape)
    norm_epsilon = 1e-3
    norm_decay = 0.99
    anchors_path = 'F:\PythonProject\pythonProject\pythonProject\CV\深度学习\第十五次课yoloV3和人脸识别网络\model_data\yolo_anchors.txt'
    classes_path = 'F:\PythonProject\pythonProject\pythonProject\CV\深度学习\第十五次课yoloV3和人脸识别网络\model_data\coco_classes.txt'
    model = yolo3(norm_epsilon, norm_decay, anchors_path, classes_path, pre_train=False)

    out = model.yolo_inference(inputs=inputs, num_anchors=len(model.anchors), num_classes=len(model.classes))
    print(out)
