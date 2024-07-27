"""
定义神经网络模型
"""
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization

def AlexNet_BN(input_shape=(224, 224, 3), output_dim=2):
    '''AlexNet + BatchNormal'''

    model = Sequential()

    # 1-1: Conv2D output shape:	 (batch_size, 54, 54, 48)
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11,11),
            strides=(4,4),
            padding="valid",
            activation="relu",
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())

    # 1-2: MaxPooling2D output shape:	 (batch_size, 26, 26, 48)
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid",
        )
    )

    # 2-1: Conv2D output shape:	 (batch_size, 26, 26, 128)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())

    # 2-2: MaxPooling2D output shape:	 (batch_size, 12, 12, 128)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid",
        )
    )

    # 3-1: Conv2D output shape:	 (batch_size, 12, 12, 192)
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )

    # 4-1: Conv2D output shape:	 (batch_size, 12, 12, 192)
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )

    # 5-1: Conv2D output shape:	 (batch_size, 12, 12, 128)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )

    # 5-2: MaxPooling2D output shape:	 (batch_size, 5, 5, 128)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid",
        )
    )

    # 铺平数据: Flatten output shape:	 (batch_size, 3200)
    model.add(Flatten())

    # 6-1: Dense output shape:	 (batch_size, 1024)
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(rate=0.25)) # 使用dropout减轻过拟合


    # 7-1: Dense output shape:	 (batch_size, 1024)
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(rate=0.25))  # 使用dropout减轻过拟合

    # 8-1: Dense output shape:	 (batch_size, 10)
    model.add( Dense(units=output_dim, activation="softmax") )

    return model
