from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()
    # 这里为了加快训练时间，把模型所有通道数减半
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的shape为(55,55,48)；
    model.add(Conv2D(
        filters=48,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='valid',
        input_shape=input_shape,
        activation='relu'
        )
    )
    model.add(BatchNormalization())  # 标准化

    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,48)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的shape为(27,27,128)；
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,128)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的shape为(13,13,192)；
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的shape为(13,13,192)；
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的shape为(13,13,128)；
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,128)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
