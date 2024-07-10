from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

import cv2
import numpy as np
import matplotlib.pyplot as plt

RGB_img = cv2.imread('three1.jpg')

RGB_img1 = np.zeros((28, 28), RGB_img.dtype)
for i in range(28):
    for j in range(28):
        if RGB_img[i][j][0] == 255:
            RGB_img1[i][j] = 0
        elif RGB_img[i][j][0] < 255:
            RGB_img1[i][j] = 255 - RGB_img[i][j][0]

plt.imshow(RGB_img1, cmap=plt.cm.binary)
plt.show()
test_images = RGB_img1.reshape((1, 28 * 28))
res = network.predict(test_images)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break