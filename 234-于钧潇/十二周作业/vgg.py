from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

def VGG(input_shape=(224,224,3), output_shape=1000):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_shape, activation='softmax'))

    return model