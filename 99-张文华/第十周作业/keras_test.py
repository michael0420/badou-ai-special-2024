'''
使用keras，实现简单的神经网络
'''

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cv2
import numpy as np


# 下载数据集
(train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.mnist.load_data()
print(f'训练数据集：\n{train_imgs}')
print(f'训练数据标签：\n{train_labels}')
print(train_imgs.shape)

# 看看测试集的第一张图片
digit = test_imgs[0]
plt.imshow(digit, cmap=plt.cm.binary)

# 预处理数据，即将数据进行归一化
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0
print(train_imgs.dtype)

# 标签进行one-hot编码，使用to_categorical
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)


# 搭建神经网络模型
network = keras.models.Sequential()
network.add(keras.layers.Flatten(input_shape=(28, 28)))
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# 训练
network.fit(train_imgs, train_labels, epochs=5, batch_size=128)

# 测试
test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 推理
# img = cv2.imread('R-C.png', 0)
# img_re = cv2.resize(img, (28, 28))
# plt.imshow(img_re, cmap=plt.cm.binary)
img_re = digit
img_re = img_re[np.newaxis, :]
res = network.predict(img_re)
print(res)
# for i in range(res[0].shape[0]):
#     if res[0, i] == 1:
#         print("the number for the picture is : ", i)
#         break
i = np.argmax(res[0])
print("the number for the picture is : ", i)
plt.show()

