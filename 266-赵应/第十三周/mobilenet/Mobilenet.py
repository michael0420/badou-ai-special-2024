# -------------------------------------------------------------#
#   MobileNet的网络部分
# -------------------------------------------------------------#
from keras.backend import relu
from keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D, \
    Reshape, Dropout
from keras.models import Model


def relu6(x):
    return relu(x, max_value=6)


def _conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, use_bias=False, padding='same')(inputs)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def _depthwise_conv_block(inputs, pointwize_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    # 3x3深度卷积
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides, use_bias=False,
                        name="conv_dw_%d" % block_id)(inputs)
    x = BatchNormalization(name="conv_dw_bn_%d" % block_id)(x)
    x = Activation(relu6, name="conv_dw_relu_%d" % block_id)(x)

    # 1x1点卷积
    x = Conv2D(pointwize_conv_filters, (1, 1), padding='same', strides=strides, use_bias=False,
               name="conv_pw_%d" % block_id)(x)
    x = BatchNormalization(name="conv_pw_bn_%d" % block_id)(x)
    return Activation(relu6, name="conv_pw_relu_%d" % block_id)(x)


def create_mobilenet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    img_input = Input(input_shape)
    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128# 56,56,128 -> 56,56,128
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
    x = Dropout(dropout)(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax')(x)
    x = Reshape((classes,))(x)

    return Model(img_input, x, name='mobilenet_1_0_224_tf')
