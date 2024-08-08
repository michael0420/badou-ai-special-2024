
import numpy as np
import tensorflow as tf
import os


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
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

    #---------------------------------------#
    #   获取种类和先验框
    #---------------------------------------#
    def _get_class(self):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            获取anchors
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):

        bn_layer = tf.layers.batch_normalization(
            inputs = input_layer,
            momentum=norm_decay, epsilon=norm_epsilon, center=True,
            scale=True, training=training, name=name)

        return  tf.nn.leaky_relu(bn_layer,alpha = 0.1)


    def _conv2d_layer(self,inputs,filters_num,kernel_size,name,use_bais = False,strides = 1):
        '''

        使用 tf.layers.conv2d 减少权重和偏执矩阵初始化过程，以及卷积后加上偏置项的操作



        :param inputs:
        :param filters_num:
        :param kernel_size:
        :param name:
        :param use_bais:
        :param strides:
        :return:
        '''
        conv = tf.layers.conv2d(
            inputs = inputs,
            filters_num = filters_num,
            kernel_size = kernel_size,
            strides = [strides,strides],
            kernel_initializer = tf.glorot_uniform_initializer(),
            padding= ('SAME' if strides==1 else 'VALID'),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4),
            use_bais = use_bais

        )

        return conv

    def _Residual_block(self,inputs,filters_num,blocks_num,kernel_size,conv_index,training=True,norm_decay=0.99,norm_epsilon = 1e-3):
        '''

        Darknet 残差block  类似 resnet两层卷积结构
        :param inputs:
        :param filters_num:
        :param blocks_num:
        :param kernel_size:
        :param conv_index:
        :param training:
        :param norm_decay:
        :param norm_epsilon:
        :return:
        '''
        inputs = tf.pad(inputs,paddings=[[0,0],[0,1],[1,0],[0,0]],mode='CONSTANT')
        layer = self._conv2d_layer(inputs,filters_num,kernel_size=3,strides=2)
        layer = self._batch_normalization_layer(layer,training=training  , norm_decay=norm_decay , norm_epsilon=norm_epsilon,name='batch_normalization_'+str(conv_index))
        conv_index+=1

        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer,filters_num // 2,kernel_size=1,strides=1)
            layer = self._batch_normalization_layer(layer,name='batch_normalization_' + str(conv_index),training=training  , norm_decay=norm_decay , norm_epsilon=norm_epsilon)
            conv_index +=1
            layer = self._conv2d_layer(layer,filters_num,kernel_size=3)
            layer = self._batch_normalization_layer(layer,name='batch_normalization_' + str(conv_index),training=training  , norm_decay=norm_decay , norm_epsilon=norm_epsilon)
            conv_index +=1
            layer +=shortcut

        return  layer,conv_index

    def _darknet53(self,inputs,conv_index,training = True,norm_decay=0.99,norm_epsilon = 1e-3):


        with tf.variable_scope('darknet53'):

            # 输入shape 416，416，3 -》 416，416，32
            conv = self._conv2d_layer(inputs,filters_num=32,kernel_size=3,strides=1)
            conv = self._batch_normalization_layer(conv,name='batch_normalization_' + str(conv_index))
            conv_index +=1
            # 208 208 64
            conv,conv_index = self._Residual_block(conv,filters_num=64,blocks_num=1,conv_index=conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            # - 104 104 128
            conv,conv_index = self._Residual_block(conv,filters_num=128,blocks_num=2,conv_index=conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            # 52 52 256
            conv,conv_index = self._Residual_block(conv,filters_num=256,blocks_num=8,conv_index=conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)

            # route1  52 52 256
            route1 = conv

            # 26 26 512
            conv, conv_index = self._Residual_block(conv,filters_num=512,blocks_num=8,training=training,norm_epsilon=norm_epsilon,norm_decay=norm_decay,conv_index=conv_index)

            route2 = conv

            # 13 13 1024
            conv, conv_index = self._Residual_block(conv,filters_num=1024,blocks_num=4, training= training,norm_decay= 0.99, norm_epsilon=1e-3)


        return  route1,route2,conv,conv_index

        '''
        1.  第一个 进行5次卷积  第一个进行5次卷积后用于下次逆卷积
        2. 第二个是进行 5+ 2 次卷积 作为一个特征层的 卷积过程是1*1 3*3 1*1 3*3 1*1

        '''

    def _yolo_block(self,inputs,filters_num , out_filters, conv_index,training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        '''


        :param inputs:
        :param filters_num:
        :param out_filters: 最后输出层卷积数量
        :param conv_index:
        :param training:
        :param norm_decay:
        :param norm_epsilon:
        :return:
        '''
        conv = self._conv2d_layer(inputs,filters_num=filters_num,kernel_size=1,strides=1)
        conv = self._batch_normalization_layer(conv,name='batch_normalization_' + str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        conv_index +=1

        # 第
        conv = self._conv2d_layer(conv, filters_num=2*filters_num, kernel_size=3, strides=1)
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                              norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1)
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1


        conv = self._conv2d_layer(conv, filters_num=2*filters_num, kernel_size=3, strides=1)
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1)
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        # 得到 route1
        route = conv

        conv = self._conv2d_layer(conv,filters_num=2*filters_num,kernel_size=3,strides=1)
        conv = self._batch_normalization_layer(conv,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        conv_index +=1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1)
        conv_index += 1

        return route,conv,conv_index


    def yolo_inference(self,inputs,num_anchors , num_classes,training=True):
        '''

        :param inputs:
        :param num_anchors:  每个grid cell 负责检测的anchor数量
        :param num_classes:   类别数量
        :param training:
        :return:
        '''
        conv_index = 1
        # route1 = 52 52 256 route2 = 26 26 512 route3 = 13 13 1024

        conv2d_26 , conv2d_43 , conv, conv_index = self._darknet53(inputs,conv_index=conv_index, training=training,  norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)



        with tf.variable_scope('yolo'):

            # 获得第一个特征层 conv 57 13 13 512 conv 59 13 13 (3 * (80 +5))
            conv2d_57 , conv2d_59,conv_index = self._yolo_block(conv,512,num_anchors*(5+num_classes),conv_index=conv_index,training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)


            # 第二个特征层
            conv2d_60 = self._conv2d_layer(conv2d_57,filters_num=256,kernel_size=1)
            conv2d_60 = self._batch_normalization_layer(conv2d_60,name='batch_normalization' + str(conv_index),training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index +=1
            # 上采样
            up_sample = tf.image.resize_nearest_neighbor(conv2d_60,[2 * tf.shape(conv2d_60)[1],2* tf.shape(conv2d_60)[1]])

            # route 0 26 26 256
            route0 = tf.concat([up_sample,conv2d_43] , axis= -1 ,name= 'concat_result_1')
            # cov2d 65= 52 , 52 256  conv2d_67 = 26 26 255
            conv2d_65 , conv2d_67 , conv_index = self._yolo_block(route0,256,num_anchors* (num_classes+5), training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)


            # 第三个特征层
            conv2d_68 = self._conv2d_layer(conv2d_65,filters_num=128,kernel_size=1,strides=1)
            conv2d_68 = self._batch_normalization_layer(conv2d_68,name='batch_normalization_'+str(conv_index),norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index +=1

            # upsample 2   52 52 128
            up_sample_1 = tf.image.resize_nearest_neighbor(conv2d_68,[2 * tf.shape(conv2d_68)[1],2* tf.shape(conv2d_68)[1]])
            route1 = tf.concat([up_sample_1,conv2d_26],axis = -1 )

            _,conv2d_75,_ = self._yolo_block(route1,128,num_anchors* (num_classes+5), conv_index=conv_index,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)


        return  [conv2d_59,conv2d_67,conv2d_75]







