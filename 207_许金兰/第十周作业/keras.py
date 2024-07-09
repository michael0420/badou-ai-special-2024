"""
@author: 207-xujinlan
用keras实现简单神经网络
"""
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
print('train images:', train_imgs.shape)
print('train labels', train_labels.shape)
print('test images', test_imgs.shape)
print('test labels', test_labels.shape)
# 输入数据reshape
train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1] * train_imgs.shape[2])
test_imgs = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1] * test_imgs.shape[2])
print('train images reshape:', train_imgs.shape)
print('test images reshape:', test_imgs.shape)
# 标签使用onehot表示
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('train labels to categorical:', train_labels.shape)
print('test labels to categorical:', test_labels.shape)
# 构建模型结构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# 模型训练
network.fit(train_imgs, train_labels, epochs=5, batch_size=128)
# 模型评价
test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=1)
print('test loss:', test_loss)
print('test accuracy:', test_acc)
test_predict = network.predict(test_imgs)
print('true number is :', np.argmax(test_labels[0]))
for i in range(len(test_predict[0])):
    if test_predict[0][i] == 1:
        print('model predict is :', i)
