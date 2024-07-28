# coding= utf-8

'''
    用 keras 实现 Inception v3 模型
'''

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras import backend
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


def Inception_v3(input_shape=[299, 299, 3], classes=1000):
    img = Input(shape=input_shape)

    x = conv2d_bn(img, 32, 3, 3,
                  strides=(2, 2),
                  padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Inception block_1 35*35
    # part_1
    # 35 * 35 * 192 ===== 35 * 35 * 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 64+64+92+32=256 nhwc-0123
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3, branch_pool],
        axis=3,
        name='block1_part1')

    # Inception block_1 35*35
    # part_2
    # 35 * 35 * 256 ===== 35 * 35 * 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+96+64=288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3, branch_pool],
        axis=3,
        name='block1_part2')

    # Inception block_2 17*17
    # part_1
    # 35 * 35 * 288 ===== 17 * 17 * 768
    branch3x3_ = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(
        branch3x3, 96, 3, 3,
        strides=(2, 2),
        padding='valid'
    )
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = layers.concatenate(
        [branch3x3_, branch3x3, branch_pool],
        axis=3,
        name='block2_part1'
    )

    # Inception block_2 17*17
    # part_2
    # 17 * 17 * 768 ===== 17 * 17 * 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7_ = conv2d_bn(x, 128, 1, 1)
    branch7x7_ = conv2d_bn(branch7x7_, 128, 1, 7)
    branch7x7_ = conv2d_bn(branch7x7_, 192, 7, 1)
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 7, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 128, 7, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7_, branch7x7, branch_pool],
        axis=3,
        name='block2_part2'
    )

    # Inception block_2 17*17
    # part_3 & part_4
    # 17 * 17 * 768 ===== 17 * 17 * 768 ===== 17 * 17 * 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7_ = conv2d_bn(x, 160, 1, 1)
        branch7x7_ = conv2d_bn(branch7x7_, 160, 1, 7)
        branch7x7_ = conv2d_bn(branch7x7_, 192, 7, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 7, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 160, 7, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7_, branch7x7, branch_pool],
            axis=3,
            name='block2_part3_4' + str(i)
        )

    # Inception block_2 17*17
    # part_5
    # 17 * 17 * 768 ===== 17 * 17 * 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7_ = conv2d_bn(x, 192, 1 ,1)
    branch7x7_ = conv2d_bn(branch7x7_, 192, 1, 7)
    branch7x7_ = conv2d_bn(branch7x7_, 192, 7, 1)
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
            [branch1x1, branch7x7_, branch7x7, branch_pool],
            axis=3,
            name='block2_part5'
        )

    # Inception block_3 8*8
    # part_1
    # 17 * 17 * 768 ===== 8 * 8 * 1280
    branch3x3_ = conv2d_bn(x, 192, 1, 1)
    branch3x3_ = conv2d_bn(branch3x3_, 320, 3, 3, strides=(2, 2), padding='valid')
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
            [branch3x3_, branch7x7x3, branch_pool],
            axis=3,
            name='block3_part1'
        )

    # Inception block_3 8*8
    # part_2 & part_3
    # 8 * 8 * 1280 ===== 8 * 8 * 2048 ===== 8 * 8 * 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)
        branch3x3_ = conv2d_bn(x, 384, 1, 1)
        branch3x3__1 = conv2d_bn(branch3x3_, 384, 1, 3)
        branch3x3__2 = conv2d_bn(branch3x3_, 384, 3, 1)
        branch3x3_ = layers.concatenate(
            [branch3x3__1, branch3x3__2],
            axis=3,
            name='block3_part2_3' + str(i)
        )
        branch3x3 = conv2d_bn(x, 448, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 384, 3, 3)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=3
        )
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3_, branch3x3, branch_pool],
            axis=3,
            name='mixed' + str(i)
        )

    # 全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predict')(x)

    inputs = img
    model = Model(inputs, x, name='inception_v3')
    return model


def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    # 设置命名规则
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    # 封装卷积 + BN + 激活函数为一个方法
    x = Conv2D(
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name
    )(x)

    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def make_good_input(input):
    # 归一化操作
    input/= 255
    input -= 0.5
    input *= 2.
    return input


if __name__ == '__main__':
    model = Inception_v3()

    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)   # 用于在指定的位置（axis）上增加一个维度， 默认增加1维
    x = make_good_input(x)

    predict = model.predict(x)
    print('推理结果：', decode_predictions(predict))
