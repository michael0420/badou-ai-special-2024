from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer):
    """
    这是一个用于2D输入的ROI池化层的函数。
    参数：
    pool_size: int
        要使用的池化区域的大小。pool_size = 7将导致一个7x7的区域。
    num_rois: 要使用的感兴趣区域的数量

    输入形状：
        两个4D张量列表[X_img, X_roi]，形状如下：
            X_img:(1, channels, rows, cols) 如果dim_ordering='th'
        或者是一个具有形状的4D张量：
            (1, rows, cols, channels) 如果dim_ordering='tf'.
        X_roi:
            (1,num_rois,4) 感兴趣区域的列表，顺序为(x,y,w,h)

    输出形状：
    一个3D张量，形状为：
    (1, num_rois, pool_size, pool_size, channels)
    """
    def __init__(self,pool_size,num_rois,**kwargs):
        # 获取图像数据的维度顺序，可以是'channels_first'或'channels_last'
        self.dim_ordering = K.image_data_format()
        # 确保维度顺序是'channels_first'或'channels_last'
        assert self.dim_ordering in {'channels_last',
                                     'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        self.pool_size = pool_size
        self.num_rois = num_rois
        # 调用父类的初始化方法
        super(RoiPoolingConv, self).__init__(**kwargs)

    # 构建通道数
    def build(self,input_shape):
        self.nb_channels = input_shape[0][3]

    # 计算输出形状并返回  (在这个函数中，它返回一个元组，其中包含输出形状的多个维度)
    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    # 调整图像和rois大小
    def call(self, x, mask=None):
        # 进行断言，确保x长度为2
        assert(len(x) == 2)
        # x的第一个元素为图像数据，第二个为区域
        img = x[0]
        rois = x[1]

        outputs = []
        for roi_idx in range(self.num_rois):
            # 遍历每个区域中的x、y、w、h
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # 用于将每个ROI区域调整到相同的尺寸（pool_size x pool_size），然后进行池化操作
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        # 张量拼接
        final_output = K.concatenate(outputs, axis=0)
        # 重新定义形状
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # 改变维度顺序
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
