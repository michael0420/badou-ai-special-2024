from keras.models import  Sequential
from keras.layers import Dense,Conv2D,Activation,MaxPool2D,Flatten,Dropout,BatchNormalization
import numpy as np
from keras.datasets import mnist
import keras.utils as np_utils
from keras.optimizers import Adam

def AlexNet(input_shape=(224,224,3),output_shape=2):
    # 一个简单的alexnet网络用于实现猫狗两个分类
    model = Sequential()

    # 如下是Alex的网络架构
    # 原模型输出的特征层是96层， 为了加速收敛用48
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    # 批标准化，加速收敛  减均值 除方差
    model.add(BatchNormalization())
    # 池化 shape:27,27,96
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'

        )
    )

    #  第二层，为了加速收敛，原来256层改为128层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    # 池化，池化过后shape是(13,13,256)
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 第三层 由原来的384 变为192 加速训练
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 原论文输出的特征层是256层，加速训练是变成128层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 最后使用步长为2的最大池化层进行池化，输出的shape是6，6，256
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 两个全连接，最初输出是1000类，这里改为两类
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='relu'))

    return model



















