from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# 初始化模型
model = Sequential()

# 第一个卷积层
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

# 第二个卷积层
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

# 第三个卷积层
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))

# 第四个卷积层
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))

# 第五个卷积层
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

# 全连接层
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 输出层
model.add(Dense(1000))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 模型的结构和参数
model.summary()