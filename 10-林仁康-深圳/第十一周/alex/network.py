from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

INPUT_SHAPE = (224,224,3)
OUTPUT_SHAPE = 2

class network:
    net = None

    def __init__(self):
        # AlexNet
        self.net = Sequential()
        # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
        # 所建模型后输出为48特征层
        self.net.add(
            Conv2D(
                filters=48,
                kernel_size=(11, 11),
                strides=(4, 4),
                padding='valid',
                input_shape=INPUT_SHAPE,
                activation='relu'
            )
        )

        self.net.add(BatchNormalization())
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
        self.net.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
        # 所建模型后输出为128特征层
        self.net.add(
            Conv2D(
                filters=128,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='same',
                activation='relu'
            )
        )

        self.net.add(BatchNormalization())
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
        self.net.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # 所建模型后输出为192特征层
        self.net.add(
            Conv2D(
                filters=192,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            )
        )
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # 所建模型后输出为192特征层
        self.net.add(
            Conv2D(
                filters=192,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            )
        )
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
        # 所建模型后输出为128特征层
        self.net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            )
        )
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
        self.net.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        # 两个全连接层，最后输出为1000类,这里改为2类
        # 缩减为1024
        self.net.add(Flatten())
        self.net.add(Dense(1024, activation='relu'))
        self.net.add(Dropout(0.25))

        self.net.add(Dense(1024, activation='relu'))
        self.net.add(Dropout(0.25))

        self.net.add(Dense(OUTPUT_SHAPE, activation='softmax'))


if __name__ == "__main__":
    pass