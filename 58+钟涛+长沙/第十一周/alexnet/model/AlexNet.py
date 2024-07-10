from  keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

def AlexNet(input_shape=(224,224,3), output_shape=2):
    model = Sequential()

    model.add(Conv2D(
        filters=48,  #输出空间的维度，即卷积核的数量。
        kernel_size=(11,11),  # 卷积核大小
        strides=(4,4),  #步长
        padding='valid',  #  'valid' 或 'same'。'valid' 表示不填充，'same' 表示填充输入以使输出具有与原始输入相同的高度和宽度。
        input_shape=input_shape, #仅在第一层需要指定，用于定义输入数据的形状。形状应该是 (height, width, channels) 的形式。
        activation='relu'))  #激活函数，用于引入非线性。常见的激活函数包括 'relu'、'sigmoid'、'tanh' 等。

    #收敛
    model.add(BatchNormalization())

    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        padding='valid'))

    model.add(Conv2D(
        filters=128,
        kernel_size =(5,5),
        strides=(1,1),
        padding='same',
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        padding='valid'))

    model.add(Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'))

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    #平展，一维数组
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model




