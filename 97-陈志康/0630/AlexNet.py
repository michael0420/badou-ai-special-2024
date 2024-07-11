from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


# 输出为cat/dog
def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()  # 线性堆叠的模型

    # 第一层卷积层
    # 只有第一层需要显式的设定Input_shape的尺寸，二三层会自适应
    model.add(
        Conv2D(
            filters=96,  # 输出特征层96层
            kernel_size=(11, 11)  # 卷积核大小11x11
            , strides=(4, 4),  # 步长4x4
            padding='valid',  # 无填充
            input_shape=input_shape,
            activation='relu'))  # 激活函数

    model.add(BatchNormalization())  # 批标准化层，有助于加速训练并提高模型的稳定性。
    # 池化
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        ))

    # 第二层卷积,输出特征层256层
    # padding='same'的目的是在卷积操作后保持输出图像的空间尺寸与输入图像的空间尺寸相同
    model.add(
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 卷积，输出特征层384
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 卷积
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape='same', activation='relu'))

    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 全连接层
    # 展平层，将多维输入展平为一维
    model.add(Flatten())
    # 第一层fc 1024个神经元
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))  # 防止过拟合

    # 第二层fc
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 输出层
    model.add(Dense(output_shape, activation='softmax'))

    return model
