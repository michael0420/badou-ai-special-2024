# @Time: 2024/7/15 20:35
# @Author: iris
# @Email: liuhw0225@126.com
from __future__ import print_function
from __future__ import absolute_import
import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=None, depth_multiplier=1, dropout=1e-3, classes=1000):
    if input_shape is None:
        input_shape = [224, 224, 3]
    image_input = Input(shape=input_shape)
    # 224,224,3 -> 112,112,32
    x = _conv_block(image_input, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(image_input, x, name='mobilenet_1_0_224_tf')

    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    普通卷积
    :param inputs:
    :param filters:
    :param kernel:
    :param strides:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel, padding='same', use_bias=False, strides=strides, name='conv1')(
        inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, point_wise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    mobile 卷积
    :param inputs:
    :param point_wise_conv_filters:
    :param depth_multiplier:
    :param strides:
    :param block_id:
    :return:
    """
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', depth_multiplier=depth_multiplier,
                        use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(filters=point_wise_conv_filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    return x


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(data):
    data /= 255.
    data -= 0.5
    data *= 2.
    return data


if __name__ == '__main__':
    model = MobileNet()
    load_weights = b'mobilenet_1_0_224_tf.h5'
    load_weights = load_weights.decode('utf-8')
    model.load_weights(load_weights)
    img_path = '../data/elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1
