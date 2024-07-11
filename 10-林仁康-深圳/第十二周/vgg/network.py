
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

INPUT_SHAPE = (224,224,3)
OUTPUT_SHAPE = 2

class Network:
    net = None

    def __init__(self):
        self.net = Sequential()
        # 1 通道：64 ；卷积核：3x3；步长：1；填充：SAME；卷积次数：2
        # 输出尺寸：224x224x64
        self.net.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(224, 224, 3),
                activation='relu'
            )
        )
        self.net.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(224, 224, 64),
                activation='relu'
            )
        )

        # 2 max pooling：滤波器 2x2；步长：2；
        # 输出尺寸减半：112x112x64
        self.net.add(MaxPooling2D(pool_size=(2, 2),
                                  strides=(1,1),
                                  padding="VALID"))

        # 3 通道：128 ；卷积核：3x3；步长：1；填充：SAME；次数：2
        # 输出尺寸：112x112x128
        self.net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(112, 112, 64),
                activation='relu'
            )
        )
        self.net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(112, 112, 128),
                activation='relu'
            )
        )

        # 4 max pooling：滤波器 2x2；步长：2；
        # 输出尺寸减半：56x56x64
        self.net.add(MaxPooling2D(pool_size=(2, 2),
                                  strides=(1, 1),
                                  padding="VALID"))

        # 5 通道：256 ；卷积核：3x3；步长：1；填充：SAME；次数：3
        # 输出尺寸：56x56x256
        self.net.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(56, 56, 64),
                activation='relu'
            )
        )
        convLayer = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            input_shape=(56, 56, 256),
            activation='relu'
        )
        self.net.add(convLayer)
        self.net.add(convLayer)

        # 6 max pooling：滤波器 2x2；步长：2；
        # 输出尺寸减半：28x28x256
        self.net.add(MaxPooling2D(pool_size=(2, 2),
                                  strides=(1, 1),
                                  padding="VALID"))

        # 7 通道：512 ；卷积核：3x3；步长：1；填充：SAME；次数：3
        # 输出尺寸：28x28x512
        self.net.add(
            Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(28, 28, 256),
                activation='relu'
            )
        )
        convLayer = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            input_shape=(28, 28, 512),
            activation='relu'
        )
        self.net.add(convLayer)
        self.net.add(convLayer)

        # 8 max pooling：滤波器 2x2；步长：2；
        # 输出尺寸减半：14x14x512
        self.net.add(MaxPooling2D(pool_size=(2, 2),
                                  strides=(1, 1),
                                  padding="VALID"))

        # 9 通道：512 ；卷积核：3x3；步长：1；填充：SAME；次数：3
        # 输出尺寸：7x7x512
        self.net.add(
            Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                input_shape=(14, 14, 512),
                activation='relu'
            )
        )
        convLayer = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            input_shape=(7, 7, 512),
            activation='relu'
        )
        self.net.add(convLayer)
        self.net.add(convLayer)

        # 10 max pooling：滤波器 2x2；步长：2；
        # 输出尺寸减半：7x7x512
        self.net.add(MaxPooling2D(pool_size=(2, 2),
                                  strides=(1, 1),
                                  padding="VALID"))

        # 11 Flatten：数据拉平为向量，
        # 通道：1000 ；全连接；步长：1；填充：SAME；次数：3，最后一次增加softmax
        # 输出尺寸：1*225088
        self.net.add(Flatten())
        self.net.add(Dense(OUTPUT_SHAPE, activation='relu'))
        self.net.add(Dense(OUTPUT_SHAPE, activation='relu'))
        self.net.add(Dense(OUTPUT_SHAPE, activation='relu'))
        self.net.add(Dense(OUTPUT_SHAPE, activation='softmax'))

        # self.net = Sequential()
        # # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
        # # 所建模型后输出为48特征层
        # self.net.add(
        #     Conv2D(
        #         filters=48,
        #         kernel_size=(11, 11),
        #         strides=(4, 4),
        #         padding='valid',
        #         input_shape=INPUT_SHAPE,
        #         activation='relu'
        #     )
        # )
        # self.net.add(BatchNormalization())
        # # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
        # self.net.add(
        #     MaxPooling2D(
        #         pool_size=(3, 3),
        #         strides=(2, 2),
        #         padding='valid'
        #     )
        # )
        # # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
        # # 所建模型后输出为128特征层
        # self.net.add(
        #     Conv2D(
        #         filters=128,
        #         kernel_size=(5, 5),
        #         strides=(1, 1),
        #         padding='same',
        #         activation='relu'
        #     )
        # )
        #
        # self.net.add(BatchNormalization())
        # # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
        # self.net.add(
        #     MaxPooling2D(
        #         pool_size=(3, 3),
        #         strides=(2, 2),
        #         padding='valid'
        #     )
        # )
        # # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # # 所建模型后输出为192特征层
        # self.net.add(
        #     Conv2D(
        #         filters=192,
        #         kernel_size=(3, 3),
        #         strides=(1, 1),
        #         padding='same',
        #         activation='relu'
        #     )
        # )
        # # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # # 所建模型后输出为192特征层
        # self.net.add(
        #     Conv2D(
        #         filters=192,
        #         kernel_size=(3, 3),
        #         strides=(1, 1),
        #         padding='same',
        #         activation='relu'
        #     )
        # )
        # # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
        # # 所建模型后输出为128特征层
        # self.net.add(
        #     Conv2D(
        #         filters=128,
        #         kernel_size=(3, 3),
        #         strides=(1, 1),
        #         padding='same',
        #         activation='relu'
        #     )
        # )
        # # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
        # self.net.add(
        #     MaxPooling2D(
        #         pool_size=(3, 3),
        #         strides=(2, 2),
        #         padding='valid'
        #     )
        # )
        # # 两个全连接层，最后输出为1000类,这里改为2类
        # # 缩减为1024
        # self.net.add(Flatten())
        # self.net.add(Dense(1024, activation='relu'))
        # self.net.add(Dropout(0.25))
        #
        # self.net.add(Dense(1024, activation='relu'))
        # self.net.add(Dropout(0.25))
        #
        # self.net.add(Dense(OUTPUT_SHAPE, activation='softmax'))

if __name__ == "__main__":
    myNet = Network()