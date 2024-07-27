import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Conv2D,BatchNormalization,MaxPooling2D,AveragePooling2D,Input,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.applications.imagenet_utils import decode_predictions  #解码来自 ImageNet 分类任务的预测结果
from keras.preprocessing import image #该模块提供了一些用于图像预处理和加载的实用函数和类。它包括用于加载、缩放和对图像进行其他转换的函数

def conv2d_bn(x,filters,num_row,num_col,strides=(1,1),padding='same',name=None):
    '''
    con2d+BN+Relu
    :param x: 模型
    :param filters: 输出特征图数量（卷积核的个数）
    :param num_row: 卷积核行数
    :param num_col: 卷积核列数
    :param strides: 步长
    :param padding: 填充方式
    :param name: 卷积核名称
    :return: C+B+R的模型结果
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'

    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters,(num_row,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
    x = BatchNormalization(scale=False,name=bn_name)(x)  #BN,scale为True时，添加缩放因子gamma到该BN层，否则不添加。添加gamma是对BN层的变化加入缩放操作。注意，gamma一般设定为可训练参数，即trainable = True。
    x = Activation('relu',name=name)(x)
    return x

def InceptionV3(input_shape=[299,299,3],classes=1000):
    # 1、调整输入图像格式为299*299*3
    img_input = Input(shape=input_shape)

    # 2、base层，3次卷积， 1次最大池化，输出尺寸为73*73*64
    x = conv2d_bn(img_input,32,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x,32,3,3,padding='valid')
    x = conv2d_bn(x,64,3,3,padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    # 3、此处为代码结构，并非ppt模型。此处两卷积，1最大池化，输出尺寸为：35*35*192
    x = conv2d_bn(x,80,1,1,padding='valid')               # 输出尺寸 73*73*80
    x = conv2d_bn(x,192,3,3,padding='valid')              # 输出尺寸 71*71*192
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)    # 输出尺寸 35*35*192

    # 4、以下Block结构根据代码进行调整。

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256

    branch1x1 = conv2d_bn(x,64,1,1)

    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3dbl = conv2d_bn(x,64,1,1)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)

    branch_pool = AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,32,1,1)

    # 对特征进行拼接得到  64+64+96+32= 256    nhwc-0123
    x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed0')

    # --------------------------------#
    # Block1 part 2 and part 3
    # 35*35*256->35*35*288->35*35*288

    for i in range(2):
        branch1x1 = conv2d_bn(x,64,1,1)

        branch5x5 = conv2d_bn(x,48,1,1)
        branch5x5 = conv2d_bn(branch5x5,64,5,5)

        branch3x3dbl = conv2d_bn(x,64,1,1)
        branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,64,1,1)

        # 64+64+96+64 = 288
        x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed'+str(1+i))

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35*35*288->17*17*768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # --------------------------------#
    # Block2 part2
    # 17*17*768->17*17*768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # --------------------------------#
    # Block2 part3 and part4
    # 17*17*768->17*17*768->17*17*768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],axis=3,name='mixed' + str(5 + i))

    # --------------------------------#
    # Block2 part5
    # 17 * 17 * 768->17 * 17 * 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17*17*768->8*8*1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # --------------------------------#
    # Block3 part2 和 part3
    # 8*8*1280->8*8*1280
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3,name='mixed9_1' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3,name='mixed9_2' + str(i))

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))


    # 平均池化后全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes,activation='softmax',name='predictions')(x)

    # 构造模型并返回
    inputs = img_input
    model = Model(inputs,x,name='inception_v3')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ =='__main__':
    model = InceptionV3()
    model.summary()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:',decode_predictions(preds))


























