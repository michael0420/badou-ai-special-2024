import numpy as np
from keras.layers import (DepthwiseConv2D,
                          Conv2D,
                          MaxPooling2D,
                          BatchNormalization,
                          Activation,
                          AveragePooling2D,
                          Input,
                          GlobalAveragePooling2D,
                          Reshape,
                          Dropout)
from keras import backend as K, Model
import keras
from keras.applications.imagenet_utils import decode_predictions


def relu6(x):
    x = K.relu(x, max_value=6)
    return x

def conv_bn(inputs, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x

def depthwise_conv2d(inputs, filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', depth_multiplier=depth_multiplier,use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    return x

def MobileNet(input_shape=(224, 224, 3), classes=1000, depth_multiplier=1, dropout=1e-3):
    inputs = Input(shape=input_shape)
    x = conv_bn(inputs, 32, strides=(2, 2))
    x = depthwise_conv2d(x, 64, depth_multiplier=depth_multiplier, block_id=1)
    x = depthwise_conv2d(x, 128, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv2d(x, 128, depth_multiplier=depth_multiplier, block_id=3)
    x = depthwise_conv2d(x, 256, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv2d(x, 256, depth_multiplier=depth_multiplier, block_id=5)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block_id=7)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block_id=8)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block_id=9)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block_id=10)
    x = depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block_id=11)
    x = depthwise_conv2d(x, 1024, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv2d(x, 1024, depth_multiplier=depth_multiplier, block_id=13)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model.load_weights('mobilenet_1_0_224_tf.h5')
    return model

def preprocess_input(x):
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))
    img = keras.preprocessing.image.load_img('elephant.jpg', target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred = model.predict(img)
    print('Predicted:', decode_predictions(pred))
