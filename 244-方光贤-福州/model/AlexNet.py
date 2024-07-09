from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
def AlexNet(input_shape=(224,224,3),output_shape=2):
    # AlexNet
    model = Sequential()
    # 卷积层大小11*11 步长4*4 卷积核个数48 不使用非零填充 激活函数是relu 输入形状224 224 3
    model.add(
        Conv2D(
            filters=48, 
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    # 添加归一化层
    model.add(BatchNormalization())
    # 最大池化层大小3*3 步长2*2 不使用非零填充
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 卷积层大小5*5 步长1*1 卷积核个数128 使用非零填充 激活函数是relu
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(5,5), 
            strides=(1,1), 
            padding='same',
            activation='relu'
        )
    )
    # 添加归一化层
    model.add(BatchNormalization())

    # 最大池化层大小3*3 步长2*2 不使用非零填充
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 卷积层大小3*3 步长1*1 卷积核个数192 使用非零填充 激活函数是relu
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3),
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )

    # 卷积层大小3*3 步长1*1 卷积核个数192 使用非零填充 激活函数是relu
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )

    # 卷积层大小3*3 步长1*1 卷积核个数128 使用非零填充 激活函数是relu
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )
    # 最大池化层大小3*3 步长2*2 不使用非零填充
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 两个全连接层 每层添加一个droupout层 丢弃率0.25 激活函数relu
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 输出为2 激活函数为softmax得到概率值
    model.add(Dense(output_shape, activation='softmax'))

    return model