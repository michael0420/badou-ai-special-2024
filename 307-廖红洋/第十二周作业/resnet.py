# -*- coding: utf-8 -*-
"""
@File    :   resnet.py
@Time    :   2024/07/13 17:59:02
@Author  :   廖红洋 
"""

from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(
    input, kernel_size, filters, stage, block
):  # input为输入，kernel表示卷积核大小，为一个整数，filters是输出维度，需要各层对齐
    filter1, filter2, filter3 = filters
    conv_name = "res" + str(stage) + block  # 当前块名编号
    bn_name = "bn" + str(stage) + block

    # 1*1卷积进行通道合并，有利于减少参数量
    x = Conv2D(filter1, (1, 1), name=conv_name + "2a")(input)
    x = BatchNormalization(name=bn_name + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter2, kernel_size, padding="same", name=conv_name + "2b")(x)
    x = BatchNormalization(name=bn_name + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter3, (1, 1), name=conv_name + "2c")(x)
    x = BatchNormalization(name=bn_name + "2c")(x)

    x = layers.add([x, input])  # 这里进行残差连接
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    conv_name = "res" + str(stage) + block  # 当前块名编号
    bn_name = "bn" + str(stage) + block

    # 1*1卷积进行通道合并，有利于减少参数量
    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name + "2a")(input_tensor)
    x = BatchNormalization(name=bn_name + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter2, kernel_size, padding="same", name=conv_name + "2b")(x)
    x = BatchNormalization(name=bn_name + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter3, (1, 1), name=conv_name + "2c")(x)
    x = BatchNormalization(name=bn_name + "2c")(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name + "1")(
        input_tensor
    )
    shortcut = BatchNormalization(name=bn_name + "1")(shortcut)
    x = layers.add([x, shortcut])  # 这里进行残差连接
    x = Activation("relu")(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
    x = BatchNormalization(name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    x = AveragePooling2D((7, 7), name="avg_pool")(x)

    x = Flatten()(x)
    x = Dense(classes, activation="softmax", name="fc1000")(x)

    model = Model(img_input, x, name="resnet50")

    model.load_weights(
        "F:/BaiduNetdiskDownload/_12_image/code/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    )

    return model


if __name__ == "__main__":
    model = ResNet50()
    model.summary()
    # 这里用的相对路径，因为我的vscode好像有点问题
    # 不是训练脚本，因为这么多参数的模型训练起来估计很慢，所以就沿用参考代码的脚本用里面的模型做个测试
    img_path = "F:/BaiduNetdiskDownload/_12_image/code/resnet50_tf/elephant.jpg"
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print("Input image shape:", x.shape)
    preds = model.predict(x)
    print("Predicted:", decode_predictions(preds))
