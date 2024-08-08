import numpy as np
import tensorflow as tf
import os

class yolo():
    def __init__(self,norm_epsilon,norm_decay,anchors_path,classes_path,pre_train):
        """
                Introduction
                ------------
                    初始化函数
                Parameters
                ----------
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

    '''
    获取种类和先验框
    '''
    def _get_class(self):
        # 获取类别的名字 返回
        classes_path = os.path.expanduser(self.classes_path) # 展开用户路径
        with open(classes_path) as f:
            class_names = f.readlines() # 打开并读取类别文件，将每行内容存入列表
        class_names = [c.strip() for c in class_names] # 去除文件名首尾空格
        return class_names
    '''
    获取anchors
    '''
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')] # 转为浮点数
        return np.array(anchors).reshape(-1, 2) # 重塑为二维数组

    def _batch_normalization_layer(self,input_layer,name=None,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''
        对卷积层提取的feature map 进行归一化处理
        Parameters
        ----------
        input_layer : 输入的四维tensor
        name ： batch norm的名字
        training ：是否为训练过程
        norm_decay ： 在预测时计算 moving average时的衰减率
        norm_epsilon ： 方差加上极小的数，防止除以0的情况

        Returns
        -------
        bn_layer ： 归一化处理之后的feature map
        '''
        # 批归一化
        bn_layer = tf.layers.batch_normalization(
            inputs=input_layer,
            training=training,
            momentum=norm_decay,
            epsilon=norm_epsilon,
            center=True,
            scale=True,
            name=name)
        # relu激活，返回激活后的feature map
        return tf.nn.leaky_relu(bn_layer, 0.1)

    def _conv2d_layer(self,inputs,filter_num,kernel_size,name,use_bias=False,strides=1):
        '''
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        '''
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filter_num,
            kernel_size=kernel_size,
            strides=[strides,strides],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
            kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('same' if strides == 1 else 'valid'),
            use_bias=use_bias,
            name=name
        )
        # 返回卷积后的feature map
        return conv

    def _Residual_block(self,inputs,filter_num,block_num,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''
        残差block
        类似resnet的两层卷积
        1x1和3x3
        1x1降维
        Parameters
        ----------
        inputs
        filter_num
        block_num
        conv_index
        training
        norm_decay
        norm_epsilon

        Returns
        -------

        '''
        # 填充
        inputs = tf.pad(inputs,paddings=[[0,0],[1,0],[1,0],[0,0]],mode='CONSTANT')
        layer = self._conv2d_layer(inputs,filter_num,3,name="conv2d_"+str(conv_index),strides=2)
        layer = self._batch_normalization_layer(layer,name='batch_normalization'+str(conv_index),
                                                training=training,norm_decay=norm_decay,
                                                norm_epsilon=norm_epsilon)
        conv_index += 1
        # 循环创建多个残差块，将结果与shortcut相加
        for _ in range(block_num):
            shortcut = layer
            # 1x1
            layer = self._conv2d_layer(layer,filter_num//2,1,name="conv2d_"+str(conv_index),strides=1)
            layer = self._batch_normalization_layer(layer,name='batch_normalization'+str(conv_index),
                                                    training=training,norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            # 3x3
            layer = self._conv2d_layer(layer,filter_num,3,name="conv2d_"+str(conv_index),strides=1)
            layer = self._batch_normalization_layer(layer,name='batch_normalization'+str(conv_index),
                                                    training=training,norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer,conv_index

    def _dartnet53(self,inputs,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''
        构建darknet53网络结构
        Parameters
        ----------
        inputs
        conv_index
        training
        norm_decay 在预测时计算moving average的衰减率
        norm_epsilon 方差加上极小的数，防止除以0的情况

        Returns
        -------

        '''
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs,filter_num=32,kernel_size=3,name="conv2d_"+str(conv_index),strides=1)
            conv = self._batch_normalization_layer(conv,name='batch_normalization'+str(conv_index),
                                                   training=training,norm_decay=norm_decay,
                                                   norm_epsilon=norm_epsilon)
            conv_index += 1
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filter_num=64,block_num=1,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filter_num=128,block_num=2,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filter_num=256,block_num=8,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)

            route1 = conv
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filter_num=512,block_num=8,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            route2 = conv
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filter_num=1024,block_num=4,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        return route1,route2,conv,conv_index

    def _yolo_block(self,inputs,filter_num,out_filters,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''
        yolo3在darknet53提取的特征层基础上，增加框三种不同比例的feature map 的block
        提高对小物品的检测率
        Parameters
        ----------
        inputs
        filter_num
        out_filters 最后输出的卷积核数量
        conv_index
        training
        norm_decay
        norm_epsilon

        Returns
        -------
        route: 返回最后一层卷积的前一层结果
        conv：返回最后一层卷积的结果
        conv_index：conv层计数
        '''
        conv = self._conv2d_layer(inputs, filter_num=filter_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filter_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filter_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filter_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filter_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filter_num=filter_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self,inputs,num_anchors,num_classes,training=True):
        '''
        构建yolo模型结构
        Parameters
        ----------
        inputs
        num_anchors 每个grid cell负责检测的anchor数量
        num_classes
        training
        -------
        '''
        conv_index = 1
        conv2d_26,conv2d_43,conv,conv_index = self._dartnet53(inputs,conv_index=conv_index,training=training,
                                                              norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
        with tf.variable_scope("yolo"):
            # 获取第一个特征层
            conv2d_57,conv2d_59,conv_index = self._yolo_block(conv,512,num_anchors * (num_classes + 5),conv_index=conv_index,training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)

            # 第二个特征层

            conv2d_60 = self._conv2d_layer(conv2d_57, filter_num=256, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name='route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            # 获得第三个特征层
            conv2d_68 = self._conv2d_layer(conv2d_65, filter_num=128, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]