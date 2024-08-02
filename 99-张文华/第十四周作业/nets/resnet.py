'''
resnet50,用于faster-R-CNN的conv_layer部分
'''

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed, Add, Activation, Flatten
from tensorflow.keras.models import Model


# 定义identity_block
def identity_block(input_tenser, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tenser)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tenser])
    x = Activation('relu')(x)
    return x


# 定义conv_block
def conv_block(input_tenser, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tenser)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + 't1')(input_tenser)
    shortcut = BatchNormalization(name=bn_name_base + 't1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# 定义resnet50网络模型
def ResNet50(inputs):

    img_input = inputs

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), (2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=3, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='f')

    return x


# 定义identity_block_td,用于classifier_layers
def identity_block_td(input_tenser, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(
        nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a'
    )(input_tenser)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis), name=bn_name_base + '2a'
    )(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(nb_filter2, kernel_size, trainable=trainable, kernel_initializer='normal', padding='same'),
        name=conv_name_base + '2b'
    )(x)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis),name=bn_name_base + '2b'
    )(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2c'
    )(x)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis), name=bn_name_base + '2c'
    )(x)

    x = Add()[x, input_tenser]
    x = Activation('relu')(x)

    return x


# 定义conv_block_td,用于classifier_layers
def conv_block_td(input_tensor, kernel_size, filters, stage, block,
                  input_shape, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 调整通道位置
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(
        Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        input_shape=input_shape, name=conv_name_base + '2a'
    )(input_tensor)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis), name=bn_name_base + '2a'
    )(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(nb_filter2, kernel_size, padding='same', trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2b'
    )(x)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis), name=bn_name_base + '2b'
    )(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), strides=strides, kernel_initializer='normal'),
        name=conv_name_base + '2c', trainable=trainable
    )(x)
    x = TimeDistributed(
        BatchNormalization(bn_axis), name=bn_name_base + '2c'
    )(x)

    shortcut = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'),
        name=conv_name_base + 't1', trainable=trainable
    )
    shortcut = TimeDistributed(BatchNormalization(bn_axis), name=bn_name_base + 't1')(shortcut)

    x = Add()[x, shortcut]
    x = Activation('relu')(x)

    return x


# 定义classifier_layers
def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(
        x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable
    )
    x = identity_block_td(x, 3, [512, 512, 2048], stage=6, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=6, block='b', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x


if __name__ == '__main__':
    inputs = Input(shape=(600, 600, 3))
    model = ResNet50(inputs)

