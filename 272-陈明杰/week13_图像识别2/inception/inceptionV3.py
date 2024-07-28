# -------------------------------------------------------------#
#   InceptionV3的网络部分
# -------------------------------------------------------------#
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


# def conv2d_bn(x,
#               filters,
#               num_row,
#               num_col,
#               strides=(1, 1),
#               padding='same',
#               name=None):
#     if name is not None:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#     x = Conv2D(
#         filters, (num_row, num_col),
#         strides=strides,
#         padding=padding,
#         use_bias=False,
#         name=conv_name)(x)
#     x = BatchNormalization(scale=False, name=bn_name)(x)
#     x = Activation('relu', name=name)(x)
#     return x
#
#
# def InceptionV3(input_shape=[299, 299, 3],
#                 classes=1000):
#     img_input = Input(shape=input_shape)
#
#     x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
#     x = conv2d_bn(x, 32, 3, 3, padding='valid')
#     x = conv2d_bn(x, 64, 3, 3)
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#     x = conv2d_bn(x, 80, 1, 1, padding='valid')
#     x = conv2d_bn(x, 192, 3, 3, padding='valid')
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#     # --------------------------------#
#     #   Block1 35x35
#     # --------------------------------#
#     # Block1 part1
#     # 35 x 35 x 192 -> 35 x 35 x 256
#     branch1x1 = conv2d_bn(x, 64, 1, 1)
#
#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
#
#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#
#     branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
#
#     # 64+64+96+32 = 256  nhwc-0123
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=3,
#         name='mixed0')
#
#     # Block1 part2
#     # 35 x 35 x 256 -> 35 x 35 x 288
#     branch1x1 = conv2d_bn(x, 64, 1, 1)
#
#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
#
#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#
#     branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
#
#     # 64+64+96+64 = 288
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=3,
#         name='mixed1')
#
#     # Block1 part3
#     # 35 x 35 x 288 -> 35 x 35 x 288
#     branch1x1 = conv2d_bn(x, 64, 1, 1)
#
#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
#
#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#
#     branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
#
#     # 64+64+96+64 = 288
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=3,
#         name='mixed2')
#
#     # --------------------------------#
#     #   Block2 17x17
#     # --------------------------------#
#     # Block2 part1
#     # 35 x 35 x 288 -> 17 x 17 x 768
#     branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
#
#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(
#         branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
#
#     branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
#     x = layers.concatenate(
#         [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')
#
#     # Block2 part2
#     # 17 x 17 x 768 -> 17 x 17 x 768
#     branch1x1 = conv2d_bn(x, 192, 1, 1)
#
#     branch7x7 = conv2d_bn(x, 128, 1, 1)
#     branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
#     branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
#
#     branch7x7dbl = conv2d_bn(x, 128, 1, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#
#     branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#         axis=3,
#         name='mixed4')
#
#     # Block2 part3 and part4
#     # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
#     for i in range(2):
#         branch1x1 = conv2d_bn(x, 192, 1, 1)
#
#         branch7x7 = conv2d_bn(x, 160, 1, 1)
#         branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
#         branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
#
#         branch7x7dbl = conv2d_bn(x, 160, 1, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#
#         branch_pool = AveragePooling2D(
#             (3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#             axis=3,
#             name='mixed' + str(5 + i))
#
#     # Block2 part5
#     # 17 x 17 x 768 -> 17 x 17 x 768
#     branch1x1 = conv2d_bn(x, 192, 1, 1)
#
#     branch7x7 = conv2d_bn(x, 192, 1, 1)
#     branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
#     branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
#
#     branch7x7dbl = conv2d_bn(x, 192, 1, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#
#     branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#         axis=3,
#         name='mixed7')
#
#     # --------------------------------#
#     #   Block3 8x8
#     # --------------------------------#
#     # Block3 part1
#     # 17 x 17 x 768 -> 8 x 8 x 1280
#     branch3x3 = conv2d_bn(x, 192, 1, 1)
#     branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
#                           strides=(2, 2), padding='valid')
#
#     branch7x7x3 = conv2d_bn(x, 192, 1, 1)
#     branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
#     branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
#     branch7x7x3 = conv2d_bn(
#         branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
#
#     branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
#     x = layers.concatenate(
#         [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')
#
#     # Block3 part2 part3
#     # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
#     for i in range(2):
#         branch1x1 = conv2d_bn(x, 320, 1, 1)
#
#         branch3x3 = conv2d_bn(x, 384, 1, 1)
#         branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
#         branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
#         branch3x3 = layers.concatenate(
#             [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))
#
#         branch3x3dbl = conv2d_bn(x, 448, 1, 1)
#         branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
#         branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
#         branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
#         branch3x3dbl = layers.concatenate(
#             [branch3x3dbl_1, branch3x3dbl_2], axis=3)
#
#         branch_pool = AveragePooling2D(
#             (3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch3x3, branch3x3dbl, branch_pool],
#             axis=3,
#             name='mixed' + str(9 + i))
#     # 平均池化后全连接。
#     x = GlobalAveragePooling2D(name='avg_pool')(x)
#     x = Dense(classes, activation='softmax', name='predictions')(x)
#
#     inputs = img_input
#
#     model = Model(inputs, x, name='inception_v3')
#
#     return model
#
#
# def preprocess_input(x):
#     x /= 255.
#     x -= 0.5
#     x *= 2.
#     return x
#
#
# if __name__ == '__main__':
#     model = InceptionV3()
#     model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
#     # 读取图片
#     img = image.load_img('elephant.jpg', target_size=(299, 299))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     p = model.predict(img_array)
#     print(decode_predictions(p, 1))
#     pass


# 第一次
def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same'):
    # use_bias = False，这句代码一定要写
    x = Conv2D(filters, kernel_size=(num_row, num_col), strides=strides, padding=padding,
               use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    x = Activation(activation='relu')(x)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    input = Input(input_shape)
    print(input.shape)
    # newH = (原H-核大小)/步长 + 1
    # newW = (原W-核大小)/步长 + 1
    x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 64, 3, 3, strides=(1, 1))
    print(x.shape)
    # newH = (原H-核大小)/步长 + 1
    # newW = (原W-核大小)/步长 + 1
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print(x.shape)
    x = conv2d_bn(x, 80, 1, 1, strides=(1, 1), padding='valid')
    print(x.shape)
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='valid')
    print(x.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print(x.shape)

    # block1 module1
    b1_m1_branch1_1x1 = conv2d_bn(x, 64, num_row=1, num_col=1, strides=(1, 1))
    b1_m1_branch2_1x1 = conv2d_bn(x, 48, num_row=1, num_col=1, strides=(1, 1))
    b1_m1_branch2_5x5 = conv2d_bn(b1_m1_branch2_1x1, 64, 5, 5, strides=(1, 1))
    b1_m1_branch3_1x1 = conv2d_bn(x, 64, 1, 1, strides=(1, 1))
    b1_m1_branch3_3x3_1 = conv2d_bn(b1_m1_branch3_1x1, 96, 3, 3, strides=(1, 1))
    b1_m1_branch3_3x3_2 = conv2d_bn(b1_m1_branch3_3x3_1, 96, 3, 3, strides=(1, 1))
    b1_m1_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(x)
    b1_m1_branch4_1x1 = conv2d_bn(b1_m1_branch4_pool, 32, 1, 1, strides=(1, 1))
    block1_module1 = layers.concatenate([b1_m1_branch1_1x1, b1_m1_branch2_5x5,
                                         b1_m1_branch3_3x3_2, b1_m1_branch4_1x1],
                                        axis=3)
    print(f'block1_module1.shape={block1_module1.shape}')

    # block1 module2
    b1_m2_branch1_1x1 = conv2d_bn(block1_module1, 64, 1, 1, strides=(1, 1))
    b1_m2_branch2_1x1 = conv2d_bn(block1_module1, 48, 1, 1, strides=(1, 1))
    b1_m2_branch2_5x5 = conv2d_bn(b1_m2_branch2_1x1, 64, 5, 5, strides=(1, 1))
    b1_m2_branch3_1x1 = conv2d_bn(block1_module1, 64, 1, 1, strides=(1, 1))
    b1_m2_branch3_3x3_1 = conv2d_bn(b1_m2_branch3_1x1, 96, 3, 3, strides=(1, 1))
    b1_m2_branch3_3x3_2 = conv2d_bn(b1_m2_branch3_3x3_1, 96, 3, 3, strides=(1, 1))
    b1_m2_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block1_module1)
    # 这里输出的64通道曾经写成输出32通道，出现bug
    b1_m2_branch4_1x1 = conv2d_bn(b1_m2_branch4_pool, 64, 1, 1, strides=(1, 1))
    block1_module2 = layers.concatenate([b1_m2_branch1_1x1, b1_m2_branch2_5x5,
                                         b1_m2_branch3_3x3_2, b1_m2_branch4_1x1], axis=3)
    print(f'block1_module2.shape={block1_module2.shape}')

    # block1 module3
    b1_m3_branch1_1x1 = conv2d_bn(block1_module2, 64, 1, 1, strides=(1, 1))
    b1_m3_branch2_1x1 = conv2d_bn(block1_module2, 48, 1, 1, strides=(1, 1))
    b1_m3_branch2_5x5 = conv2d_bn(b1_m3_branch2_1x1, 64, 5, 5, strides=(1, 1))
    b1_m3_branch3_1x1 = conv2d_bn(block1_module2, 64, 1, 1, strides=(1, 1))
    b1_m3_branch3_3x3_1 = conv2d_bn(b1_m3_branch3_1x1, 96, 3, 3, strides=(1, 1))
    b1_m3_branch3_3x3_2 = conv2d_bn(b1_m3_branch3_3x3_1, 96, 3, 3, strides=(1, 1))
    b1_m3_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block1_module2)
    b1_m3_branch4_1x1 = conv2d_bn(b1_m3_branch4_pool, 64, 1, 1, strides=(1, 1))
    block1_module3 = layers.concatenate([b1_m3_branch1_1x1, b1_m3_branch2_5x5,
                                         b1_m3_branch3_3x3_2, b1_m3_branch4_1x1], axis=3)
    print(f'block1_module3.shape={block1_module3.shape}')

    # block2 module1
    b2_m1_branch1_3x3 = conv2d_bn(block1_module3, 384, 3, 3, strides=(2, 2),
                                  padding='valid')
    b2_m1_branch2_1x1 = conv2d_bn(block1_module3, 64, 1, 1, strides=(1, 1))
    b2_m1_branch2_3x3_1 = conv2d_bn(b2_m1_branch2_1x1, 96, 3, 3, strides=(1, 1))
    b2_m1_branch2_3x3_2 = conv2d_bn(b2_m1_branch2_3x3_1, 96, 3, 3, strides=(2, 2),
                                    padding='valid')
    b2_m1_branch3_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block1_module3)
    block2_module1 = layers.concatenate([b2_m1_branch1_3x3, b2_m1_branch2_3x3_2,
                                         b2_m1_branch3_pool], axis=3)
    print(f'block2_module1.shape={block2_module1.shape}')

    # block2 module2
    b2_m2_branch1_1x1 = conv2d_bn(block2_module1, 192, 1, 1, strides=(1, 1))
    b2_m2_branch2_1x1 = conv2d_bn(block2_module1, 128, 1, 1, strides=(1, 1))
    b2_m2_branch2_1x7 = conv2d_bn(b2_m2_branch2_1x1, 128, 1, 7, strides=(1, 1))
    b2_m2_branch2_7x1 = conv2d_bn(b2_m2_branch2_1x7, 192, 7, 1, strides=(1, 1))
    b2_m2_branch3_1x1 = conv2d_bn(block2_module1, 128, 1, 1, strides=(1, 1))
    b2_m2_branch3_7x1 = conv2d_bn(b2_m2_branch3_1x1, 128, 7, 1, strides=(1, 1))
    b2_m2_branch3_1x7 = conv2d_bn(b2_m2_branch3_7x1, 128, 1, 7, strides=(1, 1))
    b2_m2_branch3_7x1 = conv2d_bn(b2_m2_branch3_1x7, 128, 7, 1, strides=(1, 1))
    b2_m2_branch3_1x7 = conv2d_bn(b2_m2_branch3_7x1, 192, 1, 7, strides=(1, 1))
    b2_m2_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block2_module1)
    b2_m2_branch4_1x1 = conv2d_bn(b2_m2_branch4_pool, 192, 1, 1, strides=(1, 1))
    block2_module2 = layers.concatenate([b2_m2_branch1_1x1, b2_m2_branch2_7x1,
                                         b2_m2_branch3_1x7, b2_m2_branch4_1x1], axis=3)
    print(f'block2_module2.shape={block2_module2.shape}')

    # block2 module3
    b2_m3_branch1_1x1 = conv2d_bn(block2_module2, 192, 1, 1, strides=(1, 1))
    b2_m3_branch2_1x1 = conv2d_bn(block2_module2, 160, 1, 1, strides=(1, 1))
    b2_m3_branch2_1x7 = conv2d_bn(b2_m3_branch2_1x1, 160, 1, 7, strides=(1, 1))
    b2_m3_branch2_7x1 = conv2d_bn(b2_m3_branch2_1x7, 192, 7, 1, strides=(1, 1))
    b2_m3_branch3_1x1 = conv2d_bn(block2_module2, 160, 1, 1, strides=(1, 1))
    b2_m3_branch3_7x1 = conv2d_bn(b2_m3_branch3_1x1, 160, 7, 1, strides=(1, 1))
    b2_m3_branch3_1x7 = conv2d_bn(b2_m3_branch3_7x1, 160, 1, 7, strides=(1, 1))
    b2_m3_branch3_7x1 = conv2d_bn(b2_m3_branch3_1x7, 160, 7, 1, strides=(1, 1))
    b2_m3_branch3_1x7 = conv2d_bn(b2_m3_branch3_7x1, 192, 1, 7, strides=(1, 1))
    b2_m3_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block2_module2)
    b2_m3_branch4_1x1 = conv2d_bn(b2_m3_branch4_pool, 192, 1, 1, strides=(1, 1))
    block2_module3 = layers.concatenate([b2_m3_branch1_1x1, b2_m3_branch2_7x1,
                                         b2_m3_branch3_1x7, b2_m3_branch4_1x1], axis=3)
    print(f'block2_module3.shape={block2_module3.shape}')

    # block2_module4
    b2_m4_branch1_1x1 = conv2d_bn(block2_module3, 192, 1, 1, strides=(1, 1))
    b2_m4_branch2_1x1 = conv2d_bn(block2_module3, 160, 1, 1, strides=(1, 1))
    b2_m4_branch2_1x7 = conv2d_bn(b2_m4_branch2_1x1, 160, 1, 7, strides=(1, 1))
    b2_m4_branch2_7x1 = conv2d_bn(b2_m4_branch2_1x7, 192, 7, 1, strides=(1, 1))
    b2_m4_branch3_1x1 = conv2d_bn(block2_module3, 160, 1, 1, strides=(1, 1))
    b2_m4_branch3_7x1 = conv2d_bn(b2_m4_branch3_1x1, 160, 7, 1, strides=(1, 1))
    b2_m4_branch3_1x7 = conv2d_bn(b2_m4_branch3_7x1, 160, 1, 7, strides=(1, 1))
    b2_m4_branch3_7x1 = conv2d_bn(b2_m4_branch3_1x7, 160, 7, 1, strides=(1, 1))
    b2_m4_branch3_1x7 = conv2d_bn(b2_m4_branch3_7x1, 192, 1, 7, strides=(1, 1))
    b2_m4_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block2_module3)
    b2_m4_branch4_1x1 = conv2d_bn(b2_m4_branch4_pool, 192, 1, 1, strides=(1, 1))
    block2_module4 = layers.concatenate([b2_m4_branch1_1x1, b2_m4_branch2_7x1,
                                         b2_m4_branch3_1x7, b2_m4_branch4_1x1], axis=3)
    print(f'block2_module4.shape={block2_module4.shape}')

    # block2_module5
    b2_m5_branch1_1x1 = conv2d_bn(block2_module4, 192, 1, 1, strides=(1, 1))
    b2_m5_branch2_1x1 = conv2d_bn(block2_module4, 192, 1, 1, strides=(1, 1))
    b2_m5_branch2_1x7 = conv2d_bn(b2_m5_branch2_1x1, 192, 1, 7, strides=(1, 1))
    b2_m5_branch2_7x1 = conv2d_bn(b2_m5_branch2_1x7, 192, 7, 1, strides=(1, 1))
    b2_m5_branch3_1x1 = conv2d_bn(block2_module4, 192, 1, 1, strides=(1, 1))
    b2_m5_branch3_7x1 = conv2d_bn(b2_m5_branch3_1x1, 192, 7, 1, strides=(1, 1))
    b2_m5_branch3_1x7 = conv2d_bn(b2_m5_branch3_7x1, 192, 1, 7, strides=(1, 1))
    b2_m5_branch3_7x1 = conv2d_bn(b2_m5_branch3_1x7, 192, 7, 1, strides=(1, 1))
    b2_m5_branch3_1x7 = conv2d_bn(b2_m5_branch3_7x1, 192, 1, 7, strides=(1, 1))
    b2_m5_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block2_module4)
    b2_m5_branch4_1x1 = conv2d_bn(b2_m5_branch4_pool, 192, 1, 1, strides=(1, 1))
    block2_module5 = layers.concatenate([b2_m5_branch1_1x1, b2_m5_branch2_7x1,
                                         b2_m5_branch3_1x7, b2_m5_branch4_1x1], axis=3)
    print(f'block2_module5.shape={block2_module5.shape}')

    # block3_module1
    b3_m1_branch1_1x1 = conv2d_bn(block2_module5, 192, 1, 1, strides=(1, 1))
    b3_m1_branch1_3x3 = conv2d_bn(b3_m1_branch1_1x1, 320, 3, 3, strides=(2, 2),
                                  padding='valid')
    b3_m1_branch2_1x1 = conv2d_bn(block2_module5, 192, 1, 1, strides=(1, 1))
    b3_m1_branch2_1x7 = conv2d_bn(b3_m1_branch2_1x1, 192, 1, 7, strides=(1, 1))
    b3_m1_branch2_7x1 = conv2d_bn(b3_m1_branch2_1x7, 192, 7, 1, strides=(1, 1))
    b3_m1_branch2_3x3 = conv2d_bn(b3_m1_branch2_7x1, 192, 3, 3, strides=(2, 2),
                                  padding='valid')
    b3_m1_branch3_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block2_module5)
    block3_module1 = layers.concatenate([b3_m1_branch1_3x3, b3_m1_branch2_3x3,
                                         b3_m1_branch3_pool], axis=3)
    print(f'block3_module1.shape={block3_module1.shape}')

    # block3_module2
    b3_m2_branch1_1x1 = conv2d_bn(block3_module1, 320, 1, 1, strides=(1, 1))
    b3_m2_branch2_1x1 = conv2d_bn(block3_module1, 384, 1, 1, strides=(1, 1))
    b3_m2_branch2_1_1x3 = conv2d_bn(b3_m2_branch2_1x1, 384, 1, 3, strides=(1, 1))
    b3_m2_branch2_2_3x1 = conv2d_bn(b3_m2_branch2_1x1, 384, 3, 1, strides=(1, 1))
    b3_m2_branch2_concat = layers.concatenate([b3_m2_branch2_1_1x3, b3_m2_branch2_2_3x1], axis=3)
    b3_m2_branch3_1x1 = conv2d_bn(block3_module1, 448, 1, 1, strides=(1, 1))
    b3_m2_branch3_3x3 = conv2d_bn(b3_m2_branch3_1x1, 384, 3, 3, strides=(1, 1))
    b3_m2_branch3_3x3_1_1x3 = conv2d_bn(b3_m2_branch3_3x3, 384, 1, 3, strides=(1, 1))
    b3_m2_branch3_3x3_2_3x1 = conv2d_bn(b3_m2_branch3_3x3, 384, 3, 1, strides=(1, 1))
    b3_m2_branch3_concat = layers.concatenate([b3_m2_branch3_3x3_1_1x3, b3_m2_branch3_3x3_2_3x1],
                                              axis=3)
    b3_m2_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block3_module1)
    b3_m2_branch4_1x1 = conv2d_bn(b3_m2_branch4_pool, 192, 1, 1, strides=(1, 1))
    block3_module2 = layers.concatenate([b3_m2_branch1_1x1, b3_m2_branch2_concat,
                                         b3_m2_branch3_concat, b3_m2_branch4_1x1], axis=3)
    print(f'block3_module2.shape={block3_module2.shape}')

    # block3_module3
    b3_m3_branch1_1x1 = conv2d_bn(block3_module2, 320, 1, 1, strides=(1, 1))
    b3_m3_branch2_1x1 = conv2d_bn(block3_module2, 384, 1, 1, strides=(1, 1))
    b3_m3_branch2_1x1_1_1x3 = conv2d_bn(b3_m3_branch2_1x1, 384, 1, 3, strides=(1, 1))
    b3_m3_branch2_1x1_2_3x1 = conv2d_bn(b3_m3_branch2_1x1, 384, 3, 1, strides=(1, 1))
    b3_m3_branch2_concat = layers.concatenate([b3_m3_branch2_1x1_1_1x3,
                                               b3_m3_branch2_1x1_2_3x1],
                                              axis=3)
    b3_m3_branch3_1x1 = conv2d_bn(block3_module2, 448, 1, 1, strides=(1, 1))
    b3_m3_branch3_3x3 = conv2d_bn(b3_m3_branch3_1x1, 384, 3, 3, strides=(1, 1))
    b3_m3_branch3_3x3_1_1x3 = conv2d_bn(b3_m3_branch3_3x3, 384, 1, 3, strides=(1, 1))
    b3_m3_branch3_3x3_2_3x1 = conv2d_bn(b3_m3_branch3_3x3, 384, 3, 1, strides=(1, 1))
    b3_m3_branch3_concat = layers.concatenate([b3_m3_branch3_3x3_1_1x3,
                                               b3_m3_branch3_3x3_2_3x1],
                                              axis=3)
    b3_m3_branch4_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                          padding='same')(block3_module2)
    b3_m3_branch4_1x1 = conv2d_bn(b3_m3_branch4_pool, 192, 1, 1, strides=(1, 1))
    block3_module3 = layers.concatenate([b3_m3_branch1_1x1, b3_m3_branch2_concat,
                                         b3_m3_branch3_concat, b3_m3_branch4_1x1], axis=3)
    print(f'block3_module3.shape={block3_module3.shape}')

    # GlobalAveragePooling2D()函数相当于 拍扁+全连接
    pool = GlobalAveragePooling2D()(block3_module3)
    net = Dense(classes, activation='softmax')(pool)
    print(net.shape)

    model = Model(input, net)
    return model


def preprocess_input(x):
    x = x / 255.
    x = x - 0.5
    x = x * 2.
    return x


# if __name__ == '__main__':
#     model = InceptionV3()
#
#     model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
#
#     img_path = 'elephant.jpg'
#     img = image.load_img(img_path, target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     x = preprocess_input(x)
#
#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds))

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights("./inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    img = image.load_img('./elephant.jpg', target_size=(299, 299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    print(decode_predictions(preds, 1))
