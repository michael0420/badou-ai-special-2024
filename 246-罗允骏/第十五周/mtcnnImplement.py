import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def P_Net():
    model = Sequential([
        Conv2D(10, (3, 3), activation='relu', padding='valid', input_shape=(None, None, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(16, (3, 3), activation='relu', padding='valid'),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # For classification of face and non-face
    ])
    return model


# 创建模型
p_net = P_Net()
p_net.summary()