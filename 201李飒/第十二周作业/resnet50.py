#--------------------------------------------------------------#
#                        Resnet 50 网络部分                     #
#--------------------------------------------------------------#
from __future__ import print_function   # python兼容性，确保Python2/3中都能使用print()函数

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers     # layers模块，用于构建网络的层
from keras.layers import Input   # 用于定义网络的输出
from keras.layers import Dense,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model  # 导入keras的Model类，用于定义和构建神经网络模型

from keras.preprocessing import image    # 用来进行图像处理
import keras.backend as K  # 导入keras的后端模块，用来处理和TensorFlow，Theano, CNTK之间的兼容性
from keras.utils.data_utils import get_file  # 用于从指定url下载文件并保存在本地
from keras.applications.imagenet_utils import  decode_predictions   # 用于将模型预测分类结果（类别索引）解码为可读标签
from keras.applications.imagenet_utils import preprocess_input    # 用于预处理输入数据（通常是图像）以符合与训练模型的输入格式和要求

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filter1, (1,1),name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(name=bn_name_base + "2a")(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding="same", name=conv_name_base + "2b")(x)
    x = BatchNormalization(name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D (filter3, (1,1),name=conv_name_base + "2c")(x)
    x = BatchNormalization(name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):

    filter1, filter2, filter3 = filters

    conv_name_base = 'res'+ str(stage) + block +"_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1,1), strides=strides,
               name=conv_name_base+"2a")(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same',
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1,1), name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    shortcut = Conv2D(filter3, (1,1), strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x

def ResNet50(input_shape=[224,224,3], classes=1000):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)     # [230,230,3]

    x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)   #[112,112,64]
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)   # [55,55,64]

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))   # [55,55,256]
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')            # [55,55,256]
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')            # [55,55,256]

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')  # [28,28,512]
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')  # [28,28,512]
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')  # [28,28,512]
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')  # [28,28,512]

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')  # [14,14,1024]
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')  # [14,14,1024]
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')  # [14,14,1024]
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')  # [14,14,1024]
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')  # [14,14,1024]
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')  # [14,14,1024]

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')  # [7,7,2048]
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')  # [7,7,2048]
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')  # [7,7,2048]

    x = AveragePooling2D((7,7),name='avg_pool')(x)   # [1,1,2048]

    x = Flatten()(x)
    x = Dense(classes, activation="softmax", name='fc1000')(x)  # [10]

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()  # 显示出模型的结构，包括每一层的名称、输出形状（尺寸）、参数数量以及整体模型参数数量等信息
    img1 =plt.imread('elephant.jpg')
    print(img1.shape)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)

    print(x.shape)
    x = preprocess_input(x)
    print('Input_shape: ', x.shape)
    preds = model.predict(x)
    print("Predicted: ", decode_predictions(preds))


