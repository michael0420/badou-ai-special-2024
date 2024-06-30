import random
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


"""
数据、标签载入及处理
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
# print(train_images.shape)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# print(train_labels[0])

"""
神经网络搭建
"""
network = models.Sequential()
network.add(layers.Dense(512, 'relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, 'softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

"""
神经网络训练
"""
network.fit(train_images, train_labels, epochs=2, batch_size=128, verbose=1)

"""
计算损失函数及准确率
"""
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

"""
神经网络推理(测试集)
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
index = random.randint(0, len(test_images) - 1)
plt.imshow(test_images[index], cmap=plt.cm.binary)
plt.show()
testimg = test_images[index].reshape([1, 28 * 28])
print(testimg.shape)
result = network.predict(testimg)
print(result.shape)
print(f'result:{result}')
for i in range(len(result[0])):
    if result[0][i] == 1:
        print(f'神经网络识别出的数字为{i}')
        break
if test_labels[index] == i:
    print('识别成功!')
else:
    print('识别失败!')

