# coding = utf-8

'''
        使用 tensorflow 实现 resnet50 模型
'''


from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import keras.backend as kbk


def identity_block(input, kernel_size, filters, stage, block):
    # identity_block分支方法
    # 一侧Conv + BatchNormalization + Activation 3次
    # 一侧无特殊处理
    f1, f2, f3 = filters

    # 设定conv和bn的命名标准
    conv_name = 'res' + str(stage) + block + '_branch'
    batch_norm_name = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(f1, (1, 1), name=conv_name + '2a')(input)
    x = BatchNormalization(name=batch_norm_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same', name=conv_name + '2b')(x)
    x = BatchNormalization(name=batch_norm_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), name=conv_name + '2c')(x)
    x = BatchNormalization(name=batch_norm_name + '2c')(x)
    x = layers.add([x, input])
    x = Activation('relu')(x)

    return x

def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    # conv_block分支方法
    # 一侧Conv + BatchNormalization + Activation 3次
    # 一侧Conv + BatchNormalization + Activation 1次
    f1, f2, f3 = filters

    # 设定conv和bn的命名标准
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(f1, (1, 1), strides=strides, name=conv_name + '2a')(input)
    x = BatchNormalization(name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same', name=conv_name + '2b')(x)
    x = BatchNormalization(name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), name=conv_name + '2c')(x)
    x = BatchNormalization(name=bn_name + '2c')(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides, name=conv_name + 'short1')(input)
    shortcut = BatchNormalization(name=bn_name + 'short1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x

def ResNet50(input_shape=[224, 224, 3], classes = 1000):
    # ResNet50:
    # input
    # ZeroPad
    # Conv2D
    # BatchNormalization
    # Activation
    # Maxpool
    # ConvBlock
    # IdentityBlock * 2
    # ConvBlock
    # IdentityBlock * 3
    # ConvBlock
    # IdentityBlock * 5
    # ConvBlock
    # IdentityBlock * 2
    # AveragePool
    # Flatten
    # FC
    # output
    img = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # 调用训练好的模型
    model = Model(img, x, name='resnet50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    return model


if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    test_img = './bike.jpg'
    img = image.load_img(test_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('输入图片尺寸：', x.shape)
    result = model.predict(x)
    print('推理结果：', decode_predictions(result))
