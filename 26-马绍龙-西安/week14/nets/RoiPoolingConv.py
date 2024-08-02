from keras.engine.topology import Layer
import keras.backend as K

# 根据所使用的后端导入相应的库
if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer):
    '''
    ROI池化层用于处理2D输入。
    该层接受一个图像张量和一个区域张量，对每个区域应用池化操作。

    # 参数
        pool_size: 池化区域的大小。
        num_rois: 一次处理的区域数量。
    # 输入形状
        包含两个4D张量的列表：图像张量和区域张量。
        图像张量形状取决于dim_ordering的设置。
        区域张量为(1, num_rois, 4)，表示(x, y, w, h)坐标。
    # 输出形状
        3D张量，形状为(1, num_rois, channels, pool_size, pool_size)。
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        '''
        初始化层的参数。
        '''
        self.dim_ordering = K.image_data_format()  # 获取图像数据的格式
        # assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        根据输入形状计算通道数。
        '''
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        '''
        计算输出形状。
        '''
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        '''
        执行ROI池化操作。

        参数:
            x: 输入张量，包含图像和区域。
            mask: 不使用，为了兼容性保留。
        '''
        assert (len(x) == 2)

        img = x[0]  # 输入图像
        rois = x[1]  # 输入区域

        outputs = []

        # 对每个区域进行池化操作
        for roi_idx in range(self.num_rois):
            # 提取每个区域的坐标
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            # 将坐标转换为整数类型
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # 对区域进行resize操作，并应用池化
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        # 将所有区域的池化结果合并为一个张量
        final_output = K.concatenate(outputs, axis=0)
        # 重塑张量以符合预期的输出形状
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # 调整维度顺序以符合'channels_last'的约定
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
