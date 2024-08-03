'''
ROI pooling layer for 2D inputs
'''

from tensorflow.keras import layers
import tensorflow.keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(layers):
    '''
    ROI pooling layer for 2D inputs
    # Arguments
        pool_size:int,Size of pooling region to use.
        num_rois:number of regions of interest to be used
    # Input shape
        list of two 4D tensor [X_img, X_roi] with shape:
        X_img:(t1, channels, rows, cols) if dim_ordering='th'
            or (t1, rows, cols, channels) if dim_ordering='tf'.
        X_roi:(t1,num_rois,4) list of rois, with ordering (x,y,w,h)
    # Output shape:
        3D tensor with shape:(t1, num_rois, channels, pool_size, pool_size)
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_data_format()
        print(self.dim_ordering)
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]
        print('nb_channels:', self.nb_channels)

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):
            x = tf.cast(rois[0, roi_idx, 0], 'int32')
            y = tf.cast(rois[0, roi_idx, 1], 'int32')
            w = tf.cast(rois[0, roi_idx, 2], 'int32')
            h = tf.cast(rois[0, roi_idx, 3], 'int32')

            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

            final_output = K.concatenate(outputs, axis=0)
            final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

            return final_output
