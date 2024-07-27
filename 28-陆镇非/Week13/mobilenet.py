from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import layers, optimizers

from keras.layers import Input, Lambda
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Reshape,Dropout
from keras.layers import Activation,BatchNormalization,Flatten,DepthwiseConv2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def MobileNet(input_shape=[32, 32, 3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=10):
    img_input = Input(shape=input_shape)

    # 32, 32, 3 -> 16,16,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 16,16,32 -> 16,16,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 16,16,64 -> 8,8,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 8,8,128 -> 8,8,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 8,8,128 -> 4,4,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)

    # 4,4,256 -> 4,4,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 4,4,256 -> 2,2,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)

    # 2,2,512 -> 2,2,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 2,2,512 -> 1,1,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 1,1,1024 -> batchSize, 1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)  #batchSize, 1,1,1024
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)  #batchSize, 1,1,10
    # x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)  #batchSize, 10

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_keras')
    # model_name = 'mobilenet_1_0_224_tf.h5'
    #     # model.load_weights(model_name)

    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


# 加载cifar-10数据集
def dataLoad():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
    # 数据集进行归一化
    train_data = train_data / 255
    test_data = test_data / 255
    # # 将标签数据集从数组类型array修改成整形类型int
    # train_label.astype(np.int)
    # test_label.astype(np.int)
    # train_data = tf.constant(train_data, dtype=tf.float64)
    # train_label = tf.constant(train_label, dtype=tf.int32)
    # test_data = tf.constant(test_data, dtype=tf.float64)
    # test_label = tf.constant(test_label, dtype=tf.int32)
    return train_data, train_label, test_data, test_label

def lossFunc(y, y_predict):
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # automaticly transfer index to one-hot, and it has already had softmax
    loss = loss_func(y, y_predict)
    return loss


if __name__ == '__main__':
    model = MobileNet()
    model.summary()
    train_data, train_label, test_data, test_label = dataLoad()
    batch_size = 128
    model.compile(loss=lossFunc,
                  optimizer=optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.fit(train_data, train_label, epochs=100, batch_size=batch_size)

    model.save_weights('last_Mobilenet.h5')
    model.load_weights("last_Mobilenet.h5")
    model.evaluate(test_data, test_label)

