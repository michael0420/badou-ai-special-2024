from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def alexnet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()
    # 第一层卷积
    model.add(Conv2D(48, (11, 11), strides=(4, 4), padding='valid', name='conv1',  input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool1'))
    # 第二层卷积
    model.add(Conv2D(128, (5, 5), padding='same',strides=(1, 1), name='conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool2'))
    # 第三层卷积
    model.add(Conv2D(192, (3, 3), padding='same', name='conv3'))
    model.add(Activation('relu'))
  #  model.add(BatchNormalization())
    # 第四层卷积
    model.add(Conv2D(192, (3, 3), padding='same', name='conv4'))
    model.add(Activation('relu'))
  #  model.add(BatchNormalization())
    # 第五层卷积
    model.add(Conv2D(128, (3, 3), padding='same', name='conv5'))
    model.add(Activation('relu'))
  #  model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool3'))
    # 全连接层
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model
