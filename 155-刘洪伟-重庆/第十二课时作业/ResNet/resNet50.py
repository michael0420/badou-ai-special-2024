# _*_ coding: UTF-8 _*_
# @Time: 2024/7/9 18:53
# @Author: iris
# @Email: liuhw0225@126.com
import numpy as np
from keras import layers
from keras.layers import Input, ZeroPadding2D, Dense, Conv2D, MaxPooling2D, \
    AveragePooling2D, Activation, BatchNormalization, Flatten
from keras.preprocessing import image
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'

    net_r = Conv2D(filters1, (1, 1), strides=strides, name=conv_base_name + '2a')(input_tensor)
    net_r = BatchNormalization(name=bn_base_name + '2a')(net_r)
    net_r = Activation('relu')(net_r)

    net_r = Conv2D(filters2, kernel_size, padding='SAME', name=conv_base_name + '2b')(net_r)
    net_r = BatchNormalization(name=bn_base_name + '2b')(net_r)
    net_r = Activation('relu')(net_r)

    net_r = Conv2D(filters3, (1, 1), name=conv_base_name + '2c')(net_r)
    net_r = BatchNormalization(name=bn_base_name + '2c')(net_r)

    y = Conv2D(filters3, (1, 1), strides=strides, name=conv_base_name + '1')(input_tensor)
    y = BatchNormalization(name=bn_base_name + '1')(y)

    net_r = layers.add([net_r, y])
    net_r = Activation('relu')(net_r)
    return net_r


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'

    net_r = Conv2D(filters1, (1, 1), name=conv_base_name + '2a')(input_tensor)
    net_r = BatchNormalization(name=bn_base_name + '2a')(net_r)
    net_r = Activation('relu')(net_r)

    net_r = Conv2D(filters2, kernel_size, padding='SAME', name=conv_base_name + '2b')(net_r)
    net_r = BatchNormalization(name=bn_base_name + '2b')(net_r)
    net_r = Activation('relu')(net_r)

    net_r = Conv2D(filters3, (1, 1), name=conv_base_name + '2c')(net_r)
    net_r = BatchNormalization(name=bn_base_name + '2c')(net_r)

    net_r = layers.add([net_r, input_tensor])
    net_r = Activation('relu')(net_r)
    return net_r


def res_net50(input_shape=None, classes=1000):
    if input_shape is None:
        input_shape = [224, 224, 3]
    image_input = Input(shape=input_shape)
    net_r = ZeroPadding2D((3, 3))(image_input)

    net_r = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(net_r)
    net_r = BatchNormalization(name='bn_conv1')(net_r)
    net_r = Activation('relu')(net_r)
    net_r = MaxPooling2D((3, 3), strides=(2, 2))(net_r)

    net_r = conv_block(net_r, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    net_r = identity_block(net_r, 3, [64, 64, 256], stage=2, block='b')
    net_r = identity_block(net_r, 3, [64, 64, 256], stage=2, block='c')

    net_r = conv_block(net_r, 3, [128, 128, 512], stage=3, block='a')
    net_r = identity_block(net_r, 3, [128, 128, 512], stage=3, block='b')
    net_r = identity_block(net_r, 3, [128, 128, 512], stage=3, block='c')
    net_r = identity_block(net_r, 3, [128, 128, 512], stage=3, block='d')

    net_r = conv_block(net_r, 3, [256, 256, 1024], stage=4, block='a')
    net_r = identity_block(net_r, 3, [256, 256, 1024], stage=4, block='b')
    net_r = identity_block(net_r, 3, [256, 256, 1024], stage=4, block='c')
    net_r = identity_block(net_r, 3, [256, 256, 1024], stage=4, block='d')
    net_r = identity_block(net_r, 3, [256, 256, 1024], stage=4, block='e')
    net_r = identity_block(net_r, 3, [256, 256, 1024], stage=4, block='f')

    net_r = conv_block(net_r, 3, [512, 512, 2048], stage=5, block='a')
    net_r = identity_block(net_r, 3, [512, 512, 2048], stage=5, block='b')
    net_r = identity_block(net_r, 3, [512, 512, 2048], stage=5, block='c')

    net_r = AveragePooling2D((7, 7), name='avg_pool')(net_r)

    net_r = Flatten()(net_r)
    net_r = Dense(classes, activation='softmax', name='fc1000')(net_r)
    models = Model(image_input, net_r, name='resnet50')
    models.load_weights("./data/resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return models


if __name__ == '__main__':
    model = res_net50()
    model.summary()
    image_path = './data/bike.jpg'
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    predicts = model.predict(x)
    print('Predicted:', decode_predictions(predicts))
