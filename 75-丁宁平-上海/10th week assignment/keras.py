# [1]
'''
将训练数据和检测数据加载到内存中（第一次运行需要下载数据，会比较慢）：
train_images是用于训练系统的手写数字图片；
train_labels适用于标注图片的信息；
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''

from tensorflow.keras.datasets import mnist
(train_images , train_labels),(test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labels',train_labels)
print('test_image.shape = ',test_images.shape)
print('test_labels',test_labels)
'''
1.train_images.shape打印结果表明，train_image是一个含有60000个元素的数组
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28
2.train_labels打印结果表明，第一张写数字图片的内容是数字5，第二张图片是数字0，以此类推。
3.train_images.shape打印结果表明，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测第一张图片的内容是7，第二张是数字2，以此类推。
'''

# [2]
'''
打印用于测试的第一张图片
'''
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

# [3]
'''
使用tensorflow.keras搭建一个有效识别图案的神经网络，
1.layers: 表述神经网络中的一个数据处理。（dense：全连接层）
2.models.Sequential(): 表示把每一个数据处理层串联起来。
3.layers.Dense(...):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“，”表示数组里的每一个元素到底包含多少个数字都没有关系
'''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# [4]
'''
把数据输入到网络模型之前，把数据归一化处理：
1.reshape(60000,28*28): train_images数组原来含有60000个元素，每个元素都是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组。
2.由于数字图像是一个灰度图，图片中每个像素点值的大小范围在0-255之间。
3.train_images.astype("float32")/255把每个像素点的值从范围在0-255转变为范围在0-1之间的浮点值。
'''

train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

# [5]
'''
把图片对应的标记也做一个更改：
目前所有图片对应的数字团是0-9.
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7.
我们需要把数值7变成一个含有十个元素的数组，然后把第八个元素设置为1，其他元素设置为0.
例如test_labels[0]的值由7变为数组[0,0,0,0,0,0,0,1,0,0]---one hot
'''

from tensorflow.keras.utils import to_categorical
print('test_labels[0] = ',test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('test_labels[0] = ',test_labels[0])

# [5]
'''
把数据输入网络进行训练：
train_images:用于训练的手写数字图片
train_labels:对应的是图片的标记
batch_size:每次网络从输入的图片数组中随机选取128个作为一组进行计算
epochs：每次计算的循环是5次
'''

network.fit(train_images,train_labels,epochs=5, batch_size=128)

# [6]
'''
测试数据输入，检验网络学习后的图片识别效果
识别效果与硬件有关（CPU/GPU）
'''

test_loss,test_acc = network.evaluate(train_images,train_labels,verbose=1)
print(test_loss)
print('test_acc',test_acc)

# [7]
'''
输入一张手写数字图片到网络中，看看它的效果
'''

(train_images,train_labels),(test_images,train_labels) = mnist.load_data()
digit1 = test_images[1]
plt.imshow(digit1,cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)


for i in range(res[1].shape[0]):
    if (res[1][i]==1):
        print('the number of picture is: ',i)
        break
