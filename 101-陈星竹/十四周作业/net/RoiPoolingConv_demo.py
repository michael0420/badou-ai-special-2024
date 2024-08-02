from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    def __init__(self,pool_size,num_rois,**kwargs):
        self.dim_ordering = K.image_data_format()
        self.num_rois = num_rois
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs) #调用父类的初始化方法

    def build(self,input_shape):
        self.nb_channels = input_shape[0][3] #获取通道数 1024

    def compute_out_shape(self,input_shape):
        # (None,num_rois,14,14,1024) 输出形状
        return None,self.num_rois,self.pool_size,self.pool_size,self.nb_channels

    def call(self,x,mask=None):
        assert(len(x) == 2) #确保输出包含两个部分：特征图和候选框Rois
        img = x[0] #特征图
        rois = x[1]
        outputs =[] #输出列表
        for roi_idx in range(self.num_rois):
            #提取每个Roi的坐标
            x = rois[0,roi_idx,0]
            y = rois[0,roi_idx,1]
            w = rois[0,roi_idx,2]
            h = rois[0,roi_idx,3]

            #改变候选框的大小
            rs = tf.image.resize_images(img[:,y:y+h,x:x+w,:],(self.pool_size,self.pool_size))

        #合并
        '''
        假设每个ROI池化后的特征图的形状是 (1, pool_size, pool_size, nb_channels)，
        如果 num_rois 是 5，
        那么合并后的张量形状将是 (5, pool_size, pool_size, nb_channels)。
        '''
        final_output = K.concatenate(outputs,axis=0)
        # 重塑形状为 (1,5,pool_size,pool_size,nb_channels)
        final_output = K.reshape(final_output,(1,self.num_rois,self.pool_size,self.pool_size,self.nb_channels))
        # K.permute_dimensions 用于置换张量的维度顺序。
        final_output = K.permute_dimensions(final_output,(0,1,2,3,4))

        return final_output