from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


#将训练数据和检测数据加载到内存中
(train_images,train_labels), (test_images,test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
#定义优化器(自适应学习率RMSProp),损失函数（多元交叉熵），监控的指标（精度）
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#在把数据输入到网络模型之前，把数据转为一维数组，并做归一化处理:
train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32')/255*0.99+0.01
test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32')/255*0.99+0.01
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
# 把数据输入网络进行训练
network.fit(train_images,train_labels,epochs=5,batch_size=128)
# 测试数据输入，检验网络学习后的图片识别效果.
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 输入一张手写数字图片到网络中，看看它的识别效果

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
#不能进行归一化处理
# test_images = test_images.astype('float32')/255*0.99+0.01
res = network.predict(test_images)
print('--->',res[1])
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break


