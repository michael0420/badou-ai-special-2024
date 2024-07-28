from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # AlexNet
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4,4),
            padding="valid",
            input_shape=input_shape,
            activation="relu"
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid"
        )
    )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same",
            activation = 'relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid"
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid"
        )
    )
    # 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    # 拍扁，改为1维
    model.add(Flatten())
    # Fc
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.25))
    # 输出结果
    model.add(Dense(output_shape, activation="softmax"))

    return model












