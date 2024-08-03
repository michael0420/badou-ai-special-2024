from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer):
    '''
    ROI pooling layer for 2D inputs.
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        '''
        在 TensorFlow 中，`K` 是 `keras.backend` ( Keras 中的一个模块，它提供了一些底层的操作和功能)的别名，`image_data_format` 是 `keras.backend` 中的一个函数，用于获取当前的图像数据格式。
        图像数据格式通常有两种：`channels_first` 和 `channels_last`。
           在 `channels_first` 格式中，图像数据的维度顺序为 `(batch_size, channels, height, width)`；
           在 `channels_last` 格式中，图像数据的维度顺序为 `(batch_size, height, width, channels)`。
        通过设置 `keras.backend.image_data_format()` 来更改默认的图像数据格式 `channels_first`
        '''
        self.dim_ordering = K.image_data_format()   # 'channels_last'
        # assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size  # 14
        self.num_rois = num_rois  # 32

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    '''
    对输入的图像张量和感兴趣区域（ROIs）张量进行处理，将每个感兴趣区域从图像中裁剪出来，并缩放为固定大小，然后将所有裁剪后的图像拼接成一个张量，并返回
    '''
    def call(self, x, mask=None):

        assert(len(x) == 2)  # 检查输入的张量 x 的长度是否为 2。如果长度不为 2，则会抛出异常

        img = x[0]  # 将输入张量 x 的第一个元素赋值给变量 img  Tensor("input_3:0", shape=(?, ?, ?, 1024), dtype=float32)
        rois = x[1]  # 将输入张量 x 的第二个元素赋值给变量 rois  # Tensor("input_2:0", shape=(?, ?, 4), dtype=float32)

        outputs = []  # 创建一个空列表 outputs，用于存储处理后的输出结果

        # 循环遍历 rois 张量的第一个维度，即 num_rois 次。 在循环内部，通过索引 roi_idx 获取 rois 张量中的坐标信息
        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]  # Tensor("roi_pooling_conv_1/strided_slice:0", shape=(), dtype=float32)
            y = rois[0, roi_idx, 1]  # Tensor("roi_pooling_conv_1/strided_slice_2:0", shape=(), dtype=float32)
            w = rois[0, roi_idx, 2]  # Tensor("roi_pooling_conv_1/strided_slice_3:0", shape=(), dtype=float32)
            h = rois[0, roi_idx, 3]  # Tensor("roi_pooling_conv_1/strided_slice_4:0", shape=(), dtype=float32)

            # 将 rois[0, roi_idx, 0] 转换为整数类型，并赋值给变量 x  Tensor("roi_pooling_conv_1/Cast:0", shape=(), dtype=int32)
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')  # Tensor("roi_pooling_conv_1/Cast_1:0", shape=(), dtype=int32)
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # 将图像 img 中以坐标 (x, y) 为左上角，宽度为 w，高度为 h 的区域缩放为大小为 (self.pool_size, self.pool_size) 的图像，并将结果存储在变量 rs 中
            #      Tensor("roi_pooling_conv_1/resize/ResizeBilinear:0", shape=(?, 14, 14, 1024), dtype=float32)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            # 将缩放后的图像添加到 outputs 列表中
            # {list:32} [<tf.Tensor 'roi_pooling_conv_1/resize/ResizeBilinear:0' shape=(?, 14, 14, 1024) dtype=float32>,
            # <tf.Tensor 'roi_pooling_conv_1/resize_1/ResizeBilinear:0' shape=(?, 14, 14, 1024) dtype=float32>,
            outputs.append(rs)

        # 将 outputs 列表中的所有图像沿着第一个维度进行拼接 : 将列表 outputs 中的所有图像拼接成一个张量，并将结果存储在变量 final_output 中。
        final_output = K.concatenate(outputs, axis=0)  # Tensor("roi_pooling_conv_1/concat:0", shape=(?, 14, 14, 1024), dtype=float32)
        # 对拼接后的张量进行形状调整 :
        #     将张量 final_output 的形状调整为 (1, num_rois, pool_size, pool_size, nb_channels)，并将结果存储在变量 final_output 中
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))  # Tensor("roi_pooling_conv_1/Reshape:0", shape=(1, 32, 14, 14, 1024), dtype=float32)
        # 对张量的维度进行重排:
        #     将张量 final_output 的维度重排为 (0, 1, 2, 3, 4)，并将结果存储在变量 final_output 中
        #     Tensor("roi_pooling_conv_1/transpose:0", shape=(1, 32, 14, 14, 1024), dtype=float32)
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
