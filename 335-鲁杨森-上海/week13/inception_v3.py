from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

def conv2d_bn(x,filters,num_row,num_col,strides=(1,1),padding='same',name=None):
    if name is not None:
        bn_name = name+'_bn'
        conv_name = name+ '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters,(num_row,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
    x = BatchNormalization(scale=False,name=bn_name)(x)
    x = Activation('relu',name=name)(x)
    return x

def InceptionV3(input_shape=[299,299,3],classes=1000):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3, padding='valid')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    '''
    Block1 35*35
    '''
    # 35*35*192 -> 35*35*256
    branch1x1 = conv2d_bn(x,64,1,1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 32+96+64+64=256 nhwc---0123
    # axis = 3 按照c来concatenate
    # 最后得到的输出通道为256
    x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed01')

    # 35*35*256 -> 35*35*288

    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+64+96 = 288
    x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed02')


    # 35*35*288 -> 35*35*288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+64+96 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed03')

    '''
    Block2 17*17
    '''
    # 35*35*288 -> 17*17*768
    #步长为2是为了让35*35 -> 17*17
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed4')

    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 单7*7卷积拆分
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 双7*7卷积拆分
    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],axis=3,name='mixed5')

    # 剩下三个
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        # 单7*7卷积拆分
        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        # 双7*7卷积拆分
        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 192+192+192+192 = 768
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed'+str(i+6))

    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed8')

    '''
    Block3 17*17*768 -> 8*8*1280
    '''

    branch3x3 = conv2d_bn(x,192,1,1)
    branch3x3 = conv2d_bn(branch3x3,320,3,3,strides=(2,2),padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x) #最大池化：8*8*768
    # 192+320+768 = 1280
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed9')

    for i in range(2):
        branch1x1 = conv2d_bn(x,320,1,1)

        branch3x3 = conv2d_bn(x,384,1,1)
        branch3x3_1 = conv2d_bn(branch3x3,384,1,3)
        branch3x3_2 = conv2d_bn(branch3x3,384,3,1)
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis=3)

        branch3x3dbl = conv2d_bn(x,448,1,1)
        branch3x3dbl = conv2d_bn(branch3x3dbl,384,3,3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl,384,1,3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl,384,3,1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1,branch3x3dbl_2],axis=3)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)
        # 384+384+384+384+320+192=2048
        x = layers.concatenate([branch1x1,branch3x3,branch3x3dbl,branch_pool],axis=3,name='mixed'+str(i+10))

    # 池化
    # 将每个特征图的所有值取平均，从而将每个特征图转化为一个单独的数值 1.降维 2.汇总特征 3.减少过拟合
    # 假设输入特征图的尺寸为 (batch_size, height, width, channels)，经过 Global Average Pooling 层后，输出的尺寸为 (batch_size, channels)
    # 相当于在进行全连接之前对数据进行排扁操作 flatten
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # 全连接
    x = Dense(classes,activation='softmax',name='predictions')(x)

    model = Model(img_input,x,name='inception_v3')

    return model

def preprocess_input(x):
    x /= 255.  # 将像素值从 [0, 255] 范围缩放到 [0, 1] 范围
    x -= 0.5   # 将像素值平移到 [-0.5, 0.5] 范围
    x *= 2.    # 将像素值缩放到 [-1, 1] 范围
    return x

if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0) # 在第0轴添加一个维度 nwhc

    # 数据预处理
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predictions:',decode_predictions(preds))