# _*_ coding: UTF-8 _*_
# @Time: 2024/6/26 19:09
# @Author: iris
# @Email: liuhw0225@126.com
from matplotlib import pyplot as plt
from tensorFlow.keras.datasets import mnist
from tensorFlow.keras.utils import to_categorical
from tensorFlow.keras import models, layers

if __name__ == '__main__':
    """获取训练数据和测试数据"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print('train_images.shape = ', train_images.shape)  # (60000, 28, 28)
    print('train_labels = ', train_labels)  # [5 0 4 ... 5 6 8]
    print('test_images.shape = ', test_images.shape)  # (10000, 28, 28)
    print('test_labels = ', test_labels)  # [7 2 1 ... 4 5 6]

    """建立神经网络"""
    network = models.Sequential()
    """隐藏层处理层（Dense->全连接）"""
    network.add(layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
    """
        输出处理层（全连接）
        softmax: 用于多分类处理
    """
    network.add(layers.Dense(10, activation='softmax'))

    """编译"""
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    """数据归一化处理"""
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    """
        ont hot 处理
        7 -- [0 0 0 0 0 0 0 1 0 0]
        2 -- [0 0 1 0 0 0 0 0 0 0]
    """
    print("before change:", test_labels[0])
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print("after change: ", test_labels[0])

    """训练"""
    network.fit(train_images, train_labels, epochs=10, batch_size=256)

    """测试"""
    test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
    print('test_loss = ', test_loss)
    print('test_acc = ', test_acc)

    """投产"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    digit = test_images[9999]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    test_images = test_images.reshape((10000, 28 * 28))
    res = network.predict(test_images)
    print(res)

    for i in range(res[9999].shape[0]):
        if (res[9999][i] == 1):
            print("the number for the picture is : ", i)
            break
