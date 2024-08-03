from keras.engine.topology import Layer
import  keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    '''

        这里roi接入两个输入 ， 一个是图像张量的输入对图像进行分类(1, rows, cols, channels)
        一个输入是对anchors 进行回归 shape是 （1，num_rois,4）

        输出是（1，num_rois,channels,pool_size,pool_size）
    '''

    def __init__(self,pool_size,num_rois,**kwargs):

        self.dim_ordering = K.image_data_format()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv,self).__init__(**kwargs)

    def built(self,input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None,self.num_rois,self.pool_size,self.pool_size,self.nb_channels

    def call(self, x, mask=None):

        assert (len(x)==2)

        img = x[0]
        # 多维张量 rois 的shape是（1，num_rois,4）

        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0,roi_idx,0]
            y = rois[0,roi_idx,1]
            w = rois[0,roi_idx,2]
            h = rois[0,roi_idx,3]

            x = K.cast(x,'int32')
            y = K.cast(y,'int32')
            w = K.cast(w,'int32')
            h = K.cast(h,'int32')

            rs = tf.image.resize_images(img[:,y:y+h,x:x+w,:],(self.pool_size,self.pool_size))
            outputs.append(rs)


        final_output = K.concatenate(outputs,axis=0)
        final_output = K.reshape(final_output,(1,self.num_rois,self.pool_size,self.pool_size,self.nb_channels))

        final_output = K.permute_dimensions(final_output,(0,1,2,3,4))

        return final_output
