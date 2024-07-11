[1]
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

[2]
'''
打印第一张图片
'''
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
'''
完成模型的搭建后，我们需要使用.compile()方法来编译模型：
优化器optimizer：该参数可指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象；
损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse，也可以为一个损失函数；
指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数。指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典。
'''
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

[4]
'''
数据预处理，二维转一维，归一化
'''
train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float32')/255
test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float32')/255

'''
标签数据处理
'''
from tensorflow.keras.utils import to_categorical
print('转换前标签：',test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('转换后标签：',test_labels[0])

[5]
'''
把数据输入网络进行训练
'''
network.fit(train_images,train_labels,epochs=5,batch_size=128)

[6]
'''
测试数据输入，检验网络学习后的图片识别效果
'''
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
#verbose：日志显示 verbose = 0 为不在标准输出流输出日志信息 verbose = 1 为输出进度条记录 verbose = 2 为每个epoch输出一行记录
print(test_loss)
print('test_acc', test_acc)

[7]
'''
输入一张图片数据，看识别效果
'''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape(10000,28*28)
predict = network.predict(test_images)
for i in range(predict[1].shape[0]):
    if predict[1][i] == 1:
        print('this num is ',i)

