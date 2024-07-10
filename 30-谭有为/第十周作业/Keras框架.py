from tensorflow.keras.datasets import mnist
import matplotlib.pylab as  plt

#加载数据集  训练图片集、标签   测试图片集、标签
(train_imgs,train_lables),(test_imgs,test_lables)=mnist.load_data()
print(train_imgs.shape,train_lables,test_imgs.shape,test_lables)

#查看测试集的第一张图片
digit=test_imgs[0]
plt.imshow(digit)
plt.show()


#使用tensorflow.Keras搭建一个有效识别图案的神经网络
#1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
#2.models.Sequential():表示把每一个数据处理层串联起来.
#3.layers.Dense(…):构造一个数据处理层。
from tensorflow.keras import models
from tensorflow.keras import layers

network=models.Sequential()
#构建隐藏层和输入层，512表示隐藏层有512个结点  输入层接收的数据格式为(（28*28）,n)
#network.add(layers.Dense(512,activation='tanh',input_shape=(28*28,)))
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
#构建输出层
network.add(layers.Dense(10,activation='softmax'))
#loss='categorical_crossentropy' 表示计算误差方法为 交叉熵
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#处理输入数据
train_imgs=train_imgs.reshape((60000,28*28))
train_imgs=train_imgs.astype('float32')/255

test_imgs=test_imgs.reshape((10000,28*28))
test_imgs=test_imgs.astype('float32')/255


#处理输出标签,to_categorical功能---把标签 7转换为数组[0,0,0,0,0,0,0,1,0,0]
from tensorflow.keras.utils import to_categorical
train_lables=to_categorical(train_lables)
test_lables=to_categorical(test_lables)

#开始训练
network.fit(train_imgs,train_lables,epochs=5,batch_size=128)

#测试集测试
#verbose是日志显示，有三个参数可选择,当verbose=0时，简单说就是不输出日志信息 ，进度条、loss、acc这些都不输出。
#当verbose=1时，带进度条的输出日志信息
test_loss,test_acc=network.evaluate(test_imgs,test_lables,verbose=1)
print('test_loss',test_loss)
print('test_acc',test_acc)

#推理
#取测试集的第二张图来试试模型是否能识别
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
digit=test_imgs[1]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

#predict函数是机器学习和深度学习模型中常用的函数之一，它可以对输入的数据进行预测或分类

test_images = test_imgs.reshape((10000, 28*28))
res=network.predict(test_imgs)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break

