[1]
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''

from tensorflow.keras.datasets import mnist  # 可以单独装keras选择不同版本，但要注意和tensorflow版本对应，避免兼容性问题
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
'''
import numpy as np  
data = np.load('mnist.npz')
# print(data.files)  # 查看文件变量名
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
print('train_image.shape = ', x_train.shape)
print('train_label = ', y_train)
print('test_image.shape = ', x_test.shape)
print('test_label = ', y_test)
'''

'''
1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28.
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''

[2]
'''
把用于测试的第一张图片打印出来看看(可省略)
'''
import matplotlib.pyplot as plt
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]
'''
使用tensorflow.keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()  # 定义network的类型是-模型（串联）
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))  # 512是第一层隐藏层节点个数
network.add(layers.Dense(10, activation='softmax'))
# 编译，除了保存模型结构还定义保存了一些额外的优化项
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# accuracy是一个整体概念，10张图对了9张就是0.9，loss是每张图片都会有的

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000, 28*28))  # 二维数组转一维
train_images = train_images.astype('float32')/255  # uint8->float32,归一化

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot独热编码
'''
from tensorflow.keras.utils import to_categorical
print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', test_labels[0])

[5]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

[6]
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_accuracy = network.evaluate(test_images, test_labels, verbose=1)  # verbose=1表示会在每个批次（batch）评估后显示进度条和当前的损失值及准确度（或其他指标）
print('test loss:', test_loss)
print('accuracy:', test_accuracy)

[7]
'''
输入一张手写数字图片到网络中，看看它的识别效果(推理)
'''
import numpy as np
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
img = test_images[2]  # 取测试集里的第三张图片做预测
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255  # 归一化之后打印最终输出值在0-1之间，否则预测结果只有0，1
res = network.predict(test_images)
# print(res[2])
# print(res[2].shape[0])
# print(res[2].shape)


max_index = np.argmax(res[2])
print("the number for the picture is : ", max_index)

# 预测结果只有0，1
'''
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
'''