'''
第十周作业
1.使用keras实现简单神经网络
2.用代码从零实现推理过程
3.使用tf实现简单神经网络

'''

# 使用keras实现简单神经网络
# 1.加载训练和测试数据
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print("train_images.shape=",train_images.shape)
print("train_labels=",train_labels)
print("test_images.shape=",test_images.shape)
print("test_labels=",test_labels)

# 打印测试数据中的第一张图，图片尺寸28*28
testimage=test_images[0]
import matplotlib.pyplot as plt
plt.imshow(testimage,cmap=plt.cm.binary)
plt.show()

# 使用tensorflow.Keras搭建一个有效识别图案的神经网络，
from tensorflow.keras import models
from tensorflow.keras import layers
'''
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
network=models.Sequential()
# 添加隐藏层
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
# 添加输出层
network.add(layers.Dense(10,activation="softmax"))
# 添加损失函数
network.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=['accuracy'])
# 数据归一化
# 将数据每个图片的二维数组转化成一维数组变成输入层
train_images=train_images.reshape((60000,28*28))
# 将每个图片的像素值从0-255转化成0-1之间方便快速计算
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255


'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''

from tensorflow.keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取150个作为一组进行计算。
epochs:每次计算的循环是6次
'''
network.fit(train_images,train_labels,batch_size=200,epochs=6)

'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss,test_acc=network.evaluate(test_images,test_labels,verbose=1)
print('test_loss=',test_loss)
print('test_acc=',test_acc)

'''
输入一张手写数字图片到网络中，看看它的识别效果
'''

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[2]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000, 28*28))
# 将训练好的模型进行推理
result=network.predict(test_images)

for i in range(result[2].shape[0]):
    if result[2][i]==1:
        print("这个图中的数字是",i)
        break




