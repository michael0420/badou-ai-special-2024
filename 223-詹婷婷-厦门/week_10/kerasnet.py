"""
[1]将训练数据和检测数据加载到内存中
"""

from tensorflow.keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print("train_images.shape = ",train_images.shape)
print("train_labels = ",train_labels)
print("test_images.shape = ",test_images.shape)
print("test_labels = ",test_labels)

digit = test_images[0]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

"""
[2]搭建神经网络
"""

from tensorflow.keras import models, layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


"""
[3]数据输入模型前，数据归一化
"""

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

"""
[4]修改图片对应的标签表示
"""
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


"""
[5]把数据输入网络进行训练
"""

network.fit(train_images, train_labels, epochs=5, batch_size=256)

"""
[6]测试数据输入，检验网络学习后的图片识别效果
"""

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_loss = ", test_loss)
print("test_acc = ", test_acc)

"""
[7]推理
"""


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
digit = digit.reshape((1, 28*28))
res = network.predict(digit)
print("res = ", res)
for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break





