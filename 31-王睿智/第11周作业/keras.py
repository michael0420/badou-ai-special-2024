
from tensorflow.keras.datasets import mnist

"""
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
"""
train_images.shape =  (60000, 28, 28)
tran_labels =  [5 0 4 ... 5 6 8]
test_images.shape =  (10000, 28, 28)
test_labels [7 2 1 ... 4 5 6]
"""
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

from tensorflow.keras import models
from tensorflow.keras import layers

# Sequential 模型结构： 层（layers）的线性堆栈
network = models.Sequential()
"""
keras.layers.Dense：
units：整数，表示该全连接层的输出维度（隐藏层节点数）。
activation：字符串，表示激活函数的名称或函数对象。默认为无激活函数。
use_bias：布尔值，表示是否使用偏置项。默认为True。
kernel_initializer：用于初始化权重矩阵的方法。默认为"glorot_uniform"。
bias_initializer：用于初始化偏置项的方法。默认为"zeros"。
kernel_regularizer：用于对权重矩阵进行正则化的方法。默认为None。
bias_regularizer：用于对偏置项进行正则化的方法。默认为None。
activity_regularizer：对层的输出进行正则化的方法。默认为None。
kernel_constraint：对权重矩阵进行约束的方法。默认为None。
bias_constraint：对偏置项进行约束的方法。默认为None。
input_shape： 输入尺寸， 28*28个输入节点
"""
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
#10：输出节点个数
network.add(layers.Dense(10, activation='softmax'))
"""
optimizer：优化器，用于控制梯度裁剪。必选项
loss：损失函数（或称目标函数、优化评分函数）。必选项
metrics：评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。评价函数和损失函数相似，只不过评价函数的结果不会用于训练过程中

"""
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
# print("train_images new shape", train_images.shape)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_lose:", test_loss)
print("test_acc:", test_acc)

'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 测试图片索引
test_number_index = 123
digit = test_images[test_number_index]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
print(test_images)
for i in range(res[1].shape[0]):
    # 比对确认图片索引为test_number_index的图是几，即判断该行中哪一列是1
    if (res[test_number_index][i] == 1):
        print("the number for the picture is : ", i)
        break
