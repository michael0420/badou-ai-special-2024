# _*_ coding: UTF-8 _*_
# @Time: 2024/7/15 18:36
# @Author: iris
# @Email: liuhw0225@126.com
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


def conv2d(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    """
    keras.Conv2D
    __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        filters：整数，表示输出空间的维度（即卷积核的数量）。它决定了卷积层的输出通道数。
        kernel_size：整数或元组，表示卷积核的大小。可以是一个整数，表示正方形卷积核的边长；也可以是一个元组，表示矩形卷积核的高和宽。
        strides：整数或元组，表示卷积核在每个维度上的步长。可以是一个整数，表示在所有维度上的相同步长；也可以是一个元组，表示在每个维度上的不同步长。
        padding：字符串，表示是否在输入的边界周围填充0值。可以取"valid"（不填充）或"same"（填充）。
        activation：字符串，表示激活函数的名称。常用的激活函数有"relu"、"sigmoid"、"tanh"等。
        input_shape：元组，表示输入数据的形状。它只需要在模型的第一层中指定，后续层会自动推断输入形状。
        data_format：字符串，表示输入数据的通道顺序。可以是"channels_last"（默认，通道维度在最后）或"channels_first"（通道维度在第二个位置）。
        dilation_rate：整数或元组，表示卷积核中的空洞率（dilated rate）。可以是一个整数，表示在所有维度上的相同空洞率；也可以是一个元组，表示在每个维度上的不同空洞率。
        use_bias：布尔值，表示是否使用偏置项。
        kernel_initializer：字符串或可调用对象，表示卷积核权重的初始化方法。
        bias_initializer：字符串或可调用对象，表示偏置项的初始化方法。
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters,
               kernel_size=(num_row, num_col),
               strides=strides,
               padding=padding,
               use_bias=False,
               name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=None, classes=1000):
    """
    定义InceptionV3网络结构
    :param input_shape: 输入像素大小
    :param classes: 分类个数
    :return:
    """
    if input_shape is None:
        input_shape = [299, 299, 3]
    image_input = Input(shape=input_shape)

    x = conv2d(image_input, 32, 3, 3, strides=(2, 2), padding='valid')  # 149 * 149 * 32
    x = conv2d(x, 32, 3, 3, padding='valid')  # 147 * 147 * 32
    x = conv2d(x, 64, 3, 3)  # 147 * 147 * 64
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # 73 * 73 * 64

    x = conv2d(x, 80, 1, 1, padding='valid')  # 71 * 71 * 80
    x = conv2d(x, 192, 3, 3, padding='valid')  # 35 * 35 * 193
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 35 * 35 * 288
    # -----------------------------------#
    #       Block1 35 * 35
    # -----------------------------------#
    # Block1 -> part1
    branch1x1 = conv2d(x, 64, 1, 1)

    branch5x5 = conv2d(x, 48, 1, 1)
    branch5x5 = conv2d(branch5x5, 64, 5, 5)

    # 两个 3*3 = 5*5
    branch5x5dbl = conv2d(x, 64, 1, 1)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 32, 1, 1)

    # 拍扁 axis = 3 按照C方向 64+64+96+32 = 256  nhwc-0123
    x = layers.concatenate([branch1x1, branch5x5, branch5x5dbl, branch_pool], axis=3, name='mixed0')

    # Block1 -> part2
    branch1x1 = conv2d(x, 64, 1, 1)

    branch5x5 = conv2d(x, 48, 1, 1)
    branch5x5 = conv2d(branch5x5, 64, 5, 5)

    # 两个 3*3 = 5*5
    branch5x5dbl = conv2d(x, 64, 1, 1)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 64, 1, 1)

    # 拍扁 axis = 3 按照C方向 64+64+96+64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch5x5dbl, branch_pool], axis=3, name='mixed1')

    # Block1 -> part3
    branch1x1 = conv2d(x, 64, 1, 1)

    branch5x5 = conv2d(x, 48, 1, 1)
    branch5x5 = conv2d(branch5x5, 64, 5, 5)

    # 两个 3*3 = 5*5
    branch5x5dbl = conv2d(x, 64, 1, 1)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 64, 1, 1)

    # 拍扁 axis = 3 按照C方向 64+64+96+64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch5x5dbl, branch_pool], axis=3, name='mixed2')

    # -----------------------------------#
    #       Block2 35 * 35
    # -----------------------------------#
    # Block2 -> part1
    branch3x3 = conv2d(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch5x5dbl = conv2d(x, 64, 1, 1)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3)
    branch5x5dbl = conv2d(branch5x5dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch5x5dbl, branch_pool], axis=3, name='mixed3')

    # Block2 -> part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d(x, 192, 1, 1)

    branch7x7 = conv2d(x, 128, 1, 1)
    branch7x7 = conv2d(branch7x7, 128, 1, 7)
    branch7x7 = conv2d(branch7x7, 192, 7, 1)

    branch9x9dbl = conv2d(x, 128, 1, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 128, 7, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 128, 1, 7)
    branch9x9dbl = conv2d(branch9x9dbl, 128, 7, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch9x9dbl, branch_pool], axis=3, name='mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d(x, 192, 1, 1)

        branch7x7 = conv2d(x, 160, 1, 1)
        branch7x7 = conv2d(branch7x7, 160, 1, 7)
        branch7x7 = conv2d(branch7x7, 192, 7, 1)

        branch9x9dbl = conv2d(x, 160, 1, 1)
        branch9x9dbl = conv2d(branch9x9dbl, 160, 7, 1)
        branch9x9dbl = conv2d(branch9x9dbl, 160, 1, 7)
        branch9x9dbl = conv2d(branch9x9dbl, 160, 7, 1)
        branch9x9dbl = conv2d(branch9x9dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch9x9dbl, branch_pool], axis=3, name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d(x, 192, 1, 1)

    branch7x7 = conv2d(x, 192, 1, 1)
    branch7x7 = conv2d(branch7x7, 192, 1, 7)
    branch7x7 = conv2d(branch7x7, 192, 7, 1)

    branch9x9dbl = conv2d(x, 192, 1, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 192, 7, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 192, 1, 7)
    branch9x9dbl = conv2d(branch9x9dbl, 192, 7, 1)
    branch9x9dbl = conv2d(branch9x9dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch9x9dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d(x, 192, 1, 1)
    branch3x3 = conv2d(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d(x, 192, 1, 1)
    branch7x7x3 = conv2d(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d(x, 320, 1, 1)

        branch3x3 = conv2d(x, 384, 1, 1)
        branch3x3_1 = conv2d(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d(x, 448, 1, 1)
        branch3x3dbl = conv2d(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))
    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(image_input, x, name='inception_v3')
    return model


def preprocess_input(data):
    data /= 255.
    data -= 0.5
    data *= 2.
    return data


if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    image_path = '../data/elephant.jpg'
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
