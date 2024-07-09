"""
用keras实现一个简单神经网络
识别手写图片
"""
[1]
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
[2]
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
[3]
from tensorflow.keras import models, layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])
[4]
# 数据的预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
[5]
# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
[6]
# 统计训练结果
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
[7]
"""输入一张手写数字图片到网络中，看看它的识别效果"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
for i in range(res[1].shape[0]):
    if(res[1][i] == 1):
        print("The number for the picture is:", i)
        break
