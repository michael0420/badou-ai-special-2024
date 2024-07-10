from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

(train_datas, train_labels), (test_datas, test_labels) = mnist.load_data()
# print(train_datas.shape, type(train_datas))
# bigimg = cv2.resize(train_datas[0], (train_datas[0].shape[0]*10, train_datas[0].shape[1]*10), interpolation=cv2.INTER_NEAREST) # cv2.INTER_LINEAR
# cv2.imshow("train", bigimg)
# cv2.waitKey()

network = models.Sequential()
network.add(layers.Dense(1024, activation='relu', input_shape=(28*28,)))
# network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#  变成1维 归一化
train_datas = train_datas.reshape((60000, 28*28))
train_datas = train_datas.astype('float32')/255
test_datas = test_datas.reshape((10000, 28*28))
test_datas = test_datas.astype('float32')/255

print("train_labels[0]:", train_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_datas, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_datas, test_labels, verbose=1)
print(test_loss, test_acc)

# 推理
img = cv2.imread('test5.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)
# 反转颜色
img = 255 - img
_, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('test', binary_image)
cv2.waitKey(0)
test_images = binary_image.reshape((1, 28*28))
res = network.predict(test_images)
print(res)
for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break
