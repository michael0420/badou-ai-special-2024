'''
练习inception-v3
'''

from tensorflow import keras
import numpy as np


# from __future__ import print_function, absolute_import


def conv2d_bn(x, filters, kernel_size, strides=(1, 1),
              padding='same', name=None):
    # 拼接层名
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None
    # 添加卷积层，归一化，激活函数
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides,
                            padding=padding, use_bias=False,
                            name=conv_name)(x)
    x = keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
    x = keras.layers.Activation('relu', name=name)(x)

    return x


# 定义模型
def inception_v3(inputs_shape=[299, 299, 3], classes=1000):
    # 实列模型输入层
    img_input = keras.layers.Input(shape=inputs_shape)

    # 模型结构隐藏层
    x = conv2d_bn(img_input, 32, (3, 3), strides=(2, 2), padding='valid')
    # x形状149, 149, 32
    x = conv2d_bn(x, 32, (3, 3), padding='valid')
    # x形状147, 147, 32
    x = conv2d_bn(x, 64, (3, 3))
    # x形状147, 147, 64
    x = keras.layers.MaxPool2D((3, 3), (2, 2))(x)
    # x形状73, 73, 64
    x = conv2d_bn(x, 80, (1, 1), padding='valid')
    # x形状73, 73, 80
    x = conv2d_bn(x, 192, (3, 3), padding='valid')
    # x形状71, 71, 192
    x = keras.layers.MaxPool2D((3, 3), (2, 2))(x)
    # x形状35, 35, 192

    ########
    # Block1_35*35
    ########
    # part 1-1
    ########
    branch1 = conv2d_bn(x, 64, (1, 1))
    # branch1.shape = 35, 35, 64

    # 使用1*1卷积降维，减少计算量
    branch5 = conv2d_bn(x, 48, (1, 1))
    branch5 = conv2d_bn(branch5, 64, (5, 5))

    # 两个3*3卷积=一个5*5卷积，但计算量减少
    branch3x3 = conv2d_bn(x, 64, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))

    # 池化
    branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, (1, 1))

    # concat
    x = keras.layers.concatenate([branch1, branch5, branch3x3, branch_pool],
                                 axis=3, name='mixed0')

    ########
    # part 1-2
    ########
    branch1 = conv2d_bn(x, 64, (1, 1))
    # branch1.shape = 35, 35, 64

    # 使用1*1卷积降维，减少计算量
    branch5 = conv2d_bn(x, 48, (1, 1))
    branch5 = conv2d_bn(branch5, 64, (5, 5))

    # 两个3*3卷积=一个5*5卷积，但计算量减少
    branch3x3 = conv2d_bn(x, 64, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))

    # 池化
    branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    # concat
    x = keras.layers.concatenate([branch1, branch5, branch3x3, branch_pool],
                                 axis=3, name='mixed1')

    ########
    # part 1-3
    ########
    branch1 = conv2d_bn(x, 64, (1, 1))
    # branch1.shape = 35, 35, 64

    # 使用1*1卷积降维，减少计算量
    branch5 = conv2d_bn(x, 48, (1, 1))
    branch5 = conv2d_bn(branch5, 64, (5, 5))

    # 两个3*3卷积=一个5*5卷积，但计算量减少
    branch3x3 = conv2d_bn(x, 64, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))
    branch3x3 = conv2d_bn(branch3x3, 96, (3, 3))

    # 池化
    branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    # concat
    x = keras.layers.concatenate([branch1, branch5, branch3x3, branch_pool],
                                 axis=3, name='mixed2')

    ########
    # Block2_17*17*768
    ########
    # part 2-1
    ########
    branch3 = conv2d_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')

    branch5 = conv2d_bn(x, 64, (1, 1))
    branch5 = conv2d_bn(branch5, 96, (3, 3))
    branch5 = conv2d_bn(branch5, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPool2D((3, 3), (2, 2))(x)

    x = keras.layers.concatenate([branch3, branch5, branch_pool],
                                 axis=3, name='mixed3')

    ########
    # part 2-2
    ########
    branch1 = conv2d_bn(x, 192, (1, 1))

    # 两个7*1， 1*7卷积等于一个7*7卷积
    branch7 = conv2d_bn(x, 128, (1, 1))
    branch7 = conv2d_bn(branch7, 128, (1, 7))
    branch7 = conv2d_bn(branch7, 192, (7, 1))

    branch9 = conv2d_bn(x, 128, (1, 1))
    branch9 = conv2d_bn(branch9, 128, (7, 1))
    branch9 = conv2d_bn(branch9, 128, (1, 7))
    branch9 = conv2d_bn(branch9, 128, (7, 1))
    branch9 = conv2d_bn(branch9, 192, (1, 7))

    branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

    x = keras.layers.concatenate([branch1, branch7, branch9, branch_pool],
                                 axis=3, name='mixed4')

    ########
    # part 2-3 and 4
    ########
    for i in range(2):
        branch1 = conv2d_bn(x, 192, (1, 1))

        branch7 = conv2d_bn(x, 160, (1, 1))
        branch7 = conv2d_bn(branch7, 160, (1, 7))
        branch7 = conv2d_bn(branch7, 192, (7, 1))

        branch9 = conv2d_bn(x, 160, (1, 1))
        branch9 = conv2d_bn(branch9, 160, (7, 1))
        branch9 = conv2d_bn(branch9, 160, (1, 7))
        branch9 = conv2d_bn(branch9, 160, (7, 1))
        branch9 = conv2d_bn(branch9, 192, (1, 7))

        branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                    padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

        x = keras.layers.concatenate([branch1, branch7, branch9, branch_pool],
                                     axis=3, name='mixed'+str(5+i))

    ########
    # part 2-5
    ########
    branch1 = conv2d_bn(x, 192, (1, 1))

    # 两个7*1， 1*7卷积等于一个7*7卷积
    branch7 = conv2d_bn(x, 192, (1, 1))
    branch7 = conv2d_bn(branch7, 192, (1, 7))
    branch7 = conv2d_bn(branch7, 192, (7, 1))

    branch9 = conv2d_bn(x, 192, (1, 1))
    branch9 = conv2d_bn(branch9, 192, (7, 1))
    branch9 = conv2d_bn(branch9, 192, (1, 7))
    branch9 = conv2d_bn(branch9, 192, (7, 1))
    branch9 = conv2d_bn(branch9, 192, (1, 7))

    branch_pool = keras.layers.AveragePooling2D((3, 3), (1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
    x = keras.layers.concatenate([branch1, branch7, branch9, branch_pool],
                                 axis=3, name='mixed7')

    ########
    # Block3_8*8*1280
    ########
    # part 3-1
    ########
    branch3 = conv2d_bn(x, 192, (1, 1))
    branch3 = conv2d_bn(branch3, 320, (3, 3), strides=(2, 2), padding='valid')

    branch9 = conv2d_bn(x, 192, (1, 1))
    branch9 = conv2d_bn(branch9, 192, (1, 7))
    branch9 = conv2d_bn(branch9, 192, (7, 1))
    branch9 = conv2d_bn(branch9, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = keras.layers.AveragePooling2D((3, 3), (2, 2))(x)

    x = keras.layers.concatenate([branch3, branch9, branch_pool],
                                 axis=3, name='mixed8')

    ########
    # part 3-2 and 3  8*8*2048
    ########
    for i in range(2):
        branch1 = conv2d_bn(x, 320, (1, 1))

        branch3 = conv2d_bn(x, 384, (1, 1))
        branch3_1 = conv2d_bn(branch3, 384, (1, 3))
        branch3_2 = conv2d_bn(branch3, 384, (3, 1))
        branch3 = keras.layers.concatenate(
            [branch3_1, branch3_2], axis=3, name='mixed9_' + str(i))

        branch5 = conv2d_bn(x, 448, (1, 1))
        branch5 = conv2d_bn(branch5, 384, (3, 3))
        branch5_1 = conv2d_bn(branch5, 384, (1, 3))
        branch5_2 = conv2d_bn(branch5, 384, (3, 1))
        branch5 = keras.layers.concatenate(
            [branch5_1, branch5_2], axis=3)

        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1),
                                                    padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

        x = keras.layers.concatenate([
            branch1, branch3, branch5, branch_pool
        ], axis=3, name='mixed'+str(9 + i))

    # 平均池化后，全连接
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dense(classes, activation='softmax',
                           name='predictions')(x)

    inputs = img_input
    model = keras.Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':

    model = inception_v3()
    # model.summary()

    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = 'elephant.jpg'
    img = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('推理结果:', keras.applications.imagenet_utils.decode_predictions(preds))


