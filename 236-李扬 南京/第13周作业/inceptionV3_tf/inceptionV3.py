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

def Conv2d_bn(X, filters, num_row, num_col, strides=(1,1), paddings='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    X = layers.Conv2D(filters=filters, kernel_size=(num_row, num_col), strides=strides, padding=paddings, use_bias=False,name=conv_name)(X)
    X = layers.BatchNormalization(scale=False, name=bn_name)(X)
    X = layers.Activation('relu', name=name)(X)
    return X

def InceptionV3(input_shape=[299,299,3], classes=1000):
    img_input = Input(shape=input_shape)

    x = Conv2d_bn(img_input, 32, 3, 3, strides=(2,2), paddings='valid')
    x = Conv2d_bn(x, 32, 3, 3, paddings='valid')
    x = Conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3,3), strides=(2, 2))(x)

    x = Conv2d_bn(x, 80, 1, 1, paddings='valid')
    x = Conv2d_bn(x, 192, 3, 3, paddings='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    #block1 part1
    brach1x1 = Conv2d_bn(x, 64, 1, 1)

    brach5x5 = Conv2d_bn(x, 48, 1, 1)
    brach5x5 = Conv2d_bn(brach5x5, 64, 5, 5)

    brach3x3db1 = Conv2d_bn(x, 64, 1, 1)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)

    brach_pool = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    brach_pool = Conv2d_bn(brach_pool, 32, 1, 1)

    x = layers.concatenate([brach1x1, brach5x5, brach3x3db1, brach_pool], axis=3, name='mixed0')

    # Block1 part2
    brach1x1 = Conv2d_bn(x, 64, 1, 1)

    brach5x5 = Conv2d_bn(x, 48, 1, 1)
    brach5x5 = Conv2d_bn(brach5x5, 64, 5, 5)

    brach3x3db1 = Conv2d_bn(x, 64, 1, 1)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)

    brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    brach_pool = Conv2d_bn(brach_pool, 64, 1, 1)
    x = layers.concatenate([brach1x1, brach5x5, brach3x3db1, brach_pool], axis=3, name='mixed1')

    # Block1 part3
    brach1x1 = Conv2d_bn(x, 64, 1, 1)

    brach5x5 = Conv2d_bn(x, 48, 1, 1)
    brach5x5 = Conv2d_bn(brach5x5, 64, 5, 5)

    brach3x3db1 = Conv2d_bn(x, 64, 1, 1)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)

    brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    brach_pool = Conv2d_bn(brach_pool, 64, 1, 1)
    x = layers.concatenate([brach1x1, brach5x5, brach3x3db1, brach_pool], axis=3, name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    branch3x3 = Conv2d_bn(x, 384, 3, 3, strides=(2, 2), paddings='valid')

    brach3x3db1 = Conv2d_bn(x, 64, 1, 1)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3)
    brach3x3db1 = Conv2d_bn(brach3x3db1, 96, 3, 3, strides=(2, 2), paddings='valid')

    brach_pool = MaxPooling2D((3,3), strides=(2,2))(x)
    x = layers.concatenate([branch3x3, brach3x3db1, brach_pool], axis=3, name='mixed3')

    # Block2 part2
    brach1x1 = Conv2d_bn(x, 192, 1, 1)

    branch7x7 = Conv2d_bn(x, 128, 1, 1)
    branch7x7 = Conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = Conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7db1 = Conv2d_bn(x, 128, 1, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 128, 7, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 128, 1, 7)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 128, 7, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 1, 7)

    brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    brach_pool = Conv2d_bn(brach_pool, 192, 1, 1)
    x = layers.concatenate([brach1x1, branch7x7, branch7x7db1, brach_pool], axis=3, name='mixed4')

    # Block2 part3 and part4
    for i in range(2):
        brach1x1 = Conv2d_bn(x, 192, 1, 1)

        branch7x7 = Conv2d_bn(x, 160, 1, 1)
        branch7x7 = Conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = Conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7db1 = Conv2d_bn(x, 160, 1, 1)
        branch7x7db1 = Conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = Conv2d_bn(branch7x7db1, 160, 1, 7)
        branch7x7db1 = Conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 1, 7)

        brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        brach_pool = Conv2d_bn(brach_pool, 192, 1, 1)
        x = layers.concatenate([brach1x1, branch7x7, branch7x7db1, brach_pool], axis=3, name='mixed' + str(5 + i))

    # Block2 part5
    brach1x1 = Conv2d_bn(x, 192, 1, 1)

    branch7x7 = Conv2d_bn(x, 192, 1, 1)
    branch7x7 = Conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = Conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7db1 = Conv2d_bn(x, 192, 1, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 7, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 1, 7)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 7, 1)
    branch7x7db1 = Conv2d_bn(branch7x7db1, 192, 1, 7)

    brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    brach_pool = Conv2d_bn(brach_pool, 192, 1, 1)
    x = layers.concatenate([brach1x1, branch7x7, branch7x7db1, brach_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    branch3x3 = Conv2d_bn(x, 192, 1, 1)
    branch3x3 = Conv2d_bn(branch3x3, 320, 3, 3, strides=(2,2), paddings='valid')

    branch7x7x3 = Conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = Conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = Conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = Conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2,2), paddings='valid')

    brach_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, brach_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    for i in range(2):
        brach1x1 = Conv2d_bn(x, 320, 1, 1)

        branch3x3 = Conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = Conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = Conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = Conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = Conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = Conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = Conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        brach_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        brach_pool = Conv2d_bn(brach_pool, 192, 1, 1)
        x = layers.concatenate([brach1x1, branch3x3, branch3x3dbl, brach_pool], axis=3, name='mixed' + str(9 + i))

    x= GlobalAveragePooling2D(name='avg_pool')(x)
    x= Dense(classes, activation='softmax', name="predictions")(x)

    inputs = img_input
    model = Model(inputs, x, name='inception_v3')
    return model

def preprocess_input(x):
    x /= 255
    x -= 0.5
    x *= 2.
    return x

if __name__ == "__main__":
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299,299))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
