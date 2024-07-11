from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten


def new_model(num_classes, rate=0.5):
    sequential = Sequential(name='VGG16')

    sequential.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
    repeat_block(sequential, 1, (3, 3), 64, (2, 2))
    repeat_block(sequential, 2, (3, 3), 128, (2, 2))
    repeat_block(sequential, 3, (3, 3), 256, (2, 2))
    repeat_block(sequential, 3, (3, 3), 512, (2, 2))
    repeat_block(sequential, 3, (3, 3), 512, (2, 2))

    sequential.add(Conv2D(4096, (7, 7), padding='VALID'))
    sequential.add(Dropout(rate))
    sequential.add(Conv2D(4096, (1, 1), padding='VALID'))
    sequential.add(Dropout(rate))
    sequential.add(Conv2D(num_classes, (1, 1), padding='VALID'))

    sequential.add(Flatten())

    return sequential


def repeat_block(sequential, repeat, kernel, filters, pool_size):
    for _ in range(0, repeat):
        sequential.add(Conv2D(filters, kernel, activation='relu', padding='same'))

    sequential.add(MaxPool2D(pool_size, strides=(2, 2), padding='valid'))


model = new_model(2)
model.summary()
