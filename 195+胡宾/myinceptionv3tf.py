# -------------------------------------------------------------#
#   InceptionV3的网络部分
# -------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import print_function

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


def folding_bn(x, filters, num_row, num_col, step=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, (num_row, num_col), strides=step, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def my_inception_v3(input_shape=[299, 299, 3], classes=1000):
    input_imag = Input(shape=input_shape)
    x = folding_bn(input_imag, 32, 3, 3, step=(2, 2), padding='valid')
    x = folding_bn(x, 32, 3, 3, step=(1, 1), padding='valid')
    x = folding_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = folding_bn(x, 80, 1, 1, padding='valid')
    x = folding_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = folding_bn(x, 64, 1, 1)

    branch1x1x5 = folding_bn(x, 48, 1, 1)
    branch5x5 = folding_bn(branch1x1x5, 64, 5, 5)

    branch3x3 = folding_bn(x, 64, 1, 1)
    branch3x3 = folding_bn(branch3x3, 96, 3, 3)
    branch3x3 = folding_bn(branch3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = folding_bn(branch_pool, 32, 1, 1)

    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='figure1')

    # Block2 part2
    # 35 x 35 x 256 -> 35 x 35 x 288

    branch_1x1 = folding_bn(x, 64, 1, 1)

    branch_1x1x5 = folding_bn(x, 48, 1, 1)
    branch_5x5 = folding_bn(branch_1x1x5, 64, 5, 5)

    branch_3x3 = folding_bn(x, 64, 1, 1)
    branch_3x3 = folding_bn(branch_3x3, 96, 3, 3)
    branch_3x3 = folding_bn(branch_3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = folding_bn(branch_pool, 64, 1, 1)

    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3, branch_pool], axis=3, name='figure2')
    # Block3 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch_1x1 = folding_bn(x, 64, 1, 1)

    branch_1x1x5 = folding_bn(x, 48, 1, 1)
    branch_5x5 = folding_bn(branch_1x1x5, 64, 5, 5)

    branch_3x3 = folding_bn(x, 64, 1, 1)
    branch_3x3 = folding_bn(branch_3x3, 96, 3, 3)
    branch_3x3 = folding_bn(branch_3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = folding_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3, branch_pool], axis=3, name='figure3')

    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch_1x1 = folding_bn(x, 384, 3, 3, step=(2, 2), padding='valid')

    branch_5x5 = folding_bn(x, 64, 1, 1)
    branch_5x5 = folding_bn(branch_5x5, 96, 3, 3)
    branch_5x5 = folding_bn(branch_5x5, 96, 3, 3, step=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch_1x1, branch_5x5, branch_pool], axis=3, name='figure4')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch_1x1 = folding_bn(x, 192, 1, 1)

    branch_7x7 = folding_bn(x, 128, 1, 1)
    branch_7x7 = folding_bn(branch_7x7, 128, 1, 7)
    branch_7x7 = folding_bn(branch_7x7, 192, 7, 1)

    branch_7x7dbl = folding_bn(x, 128, 1, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 128, 7, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 128, 1, 7)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 128, 7, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = folding_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch_1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=3, name='figure5')
    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = folding_bn(x, 192, 1, 1)

        branch_7x7 = folding_bn(x, 160, 1, 1)
        branch_7x7 = folding_bn(branch_7x7, 160, 1, 7)
        branch_7x7 = folding_bn(branch_7x7, 192, 7, 1)

        branch_7x7dbl = folding_bn(x, 160, 1, 1)
        branch_7x7dbl = folding_bn(branch_7x7dbl, 160, 7, 1)
        branch_7x7dbl = folding_bn(branch_7x7dbl, 160, 1, 7)
        branch_7x7dbl = folding_bn(branch_7x7dbl, 160, 7, 1)
        branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = folding_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=3, name='figure'+ str(i+6))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = folding_bn(x, 192, 1, 1)

    branch_7x7 = folding_bn(x, 192, 1, 1)
    branch_7x7 = folding_bn(branch_7x7, 192, 1, 7)
    branch_7x7 = folding_bn(branch_7x7, 192, 7, 1)

    branch_7x7dbl = folding_bn(x, 192, 1, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 7, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 1, 7)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 7, 1)
    branch_7x7dbl = folding_bn(branch_7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = folding_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=3, name='figure8')

    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch_3x3= folding_bn(x, 192, 1, 1)
    branch_3x3= folding_bn(branch_3x3, 320, 3, 3, step=(2, 2), padding='valid')

    branch_7x7 = folding_bn(x, 192, 1, 1)
    branch_7x7 = folding_bn(branch_7x7, 192, 1, 7)
    branch_7x7 = folding_bn(branch_7x7, 192, 7, 1)
    branch_7x7 = folding_bn(branch_7x7, 192, 3, 3, step=(2, 2), padding="valid")

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch_3x3, branch_7x7, branch_pool], axis=3, name='figure9')
    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch_1x1 = folding_bn(x, 320, 1, 1)

        branch_1x1x3 = folding_bn(x, 384, 1, 1)
        branch_3x1 = folding_bn(branch_1x1x3, 384, 1, 3)
        branch_1x3 = folding_bn(branch_1x1x3, 384, 3, 1)
        branch_1x1x3 = layers.concatenate([branch_3x1, branch_1x3], axis=3, name='figure_'+ str(i))

        branch_3x3dbl = folding_bn(x, 448, 1, 1)
        branch_3x3dbl = folding_bn(branch_3x3dbl, 384, 3, 3)
        branch_3x3dbl_1 = folding_bn(branch_3x3dbl, 384, 1, 3)
        branch_3x3dbl_2 = folding_bn(branch_3x3dbl, 384, 3, 1)
        branch_3x3dbl=layers.concatenate([branch_3x3dbl_1, branch_3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = folding_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate([branch_1x1, branch_1x1x3, branch_3x3dbl, branch_pool], axis=3, name='figure'+str(10+i))

    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = input_imag

    model = Model(inputs, x, name='inception_v3')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__== '__main__':
    model = my_inception_v3()
    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))





