'''
使用keras构建了简单的ALexNet卷积神经网络
'''

from tensorflow import keras


# 定义一个方法能够返回一个ALexNet卷积神经网络模型
def AlexNet(input_shape=(224, 224, 3), output_shape=2):

    # 初始化一个空的顺序神经网络
    model = keras.models.Sequential()

    # 为模型添加第一个卷积层，包含一次卷积，relu，最大池化
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层（为了降低计算）
    model.add(keras.layers.Conv2D(
        filters=48, kernel_size=(11, 11), strides=(4, 4),
        padding='valid', input_shape=input_shape, activation='relu'
    ))
    # 为加速收敛，做在每个批次（batch）上进行归一化
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='valid'
    ))

    # 为模型添加第二个卷积层
    model.add(keras.layers.Conv2D(
        filters=128, kernel_size=(5, 5), strides=(1, 1),
        padding='same', activation='relu'
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='valid'
    ))

    # 为模型添加第三层卷积
    model.add(keras.layers.Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu'
    ))
    model.add(keras.layers.Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu'
    ))
    model.add(keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu'
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='valid'
    ))
    # 次数输出的tensor形状为：6*6*128

    # 拍扁进入FC
    model.add(keras.layers.Flatten())

    # 构建三个全连接层
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.25)) # 防止过拟合，使用Dropout
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(output_shape, activation='softmax'))

    return model
