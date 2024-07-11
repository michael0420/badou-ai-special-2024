# from keras.dataset import mnist
#
from tensorflow.keras.datasets import mnist

# mnist.load_data(path)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('训练集图像数量:', len(train_images))
print('测试集图像数量:', len(test_images))
print('训练集标签数量:', len(train_labels))
print('训练集标签数量:', len(test_labels))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)

# design model
model = Sequential()
model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Convolution2D(50, (5, 5)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=0.001)
# compile model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# training model
# model.fit(x_train, y_train, batch_size=100, epochs=5)
model.fit(x_train, y_train, batch_size=100, epochs=5)
# test model
print(model.evaluate(x_test, y_test, batch_size=100))
# save model
# model.save('/Users/zhang001/Desktop/my_model2.h5')
model.save('path_to_my_model.h5')




