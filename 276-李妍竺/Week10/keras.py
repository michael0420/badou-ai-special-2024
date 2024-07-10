
'''
1.载入训练、检测数据
'''
from tensorflow.keras.datasets import mnist

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

print('train_images.shape = {},\ntrain_labels = {}'.format(train_images.shape,train_labels))
print('test_images.shape = {},\ntest_labels = {}'.format(test_images.shape,test_labels))

'''
2.展示测试集的第一张图片
'''
import matplotlib.pyplot as plt

number = test_images[0]
plt.imshow(number, cmap=plt.cm.gray_r)   # cm:colormap子库，binary :渲染成黑白两色
plt.show()

'''
3.使用tensorflow.Keras搭建神经网络
    1.layer:数据处理层 —— dense：全连接层     layers.Dense():构造一个数据处理层
    2.models.Sequential(): 将每一个数据处理层串联起来：输入层——隐藏层——输出层
    3.input_shape=(28*28,):规定数据的输入格式。 '，'确保了括号内被解释为一个元素，而不是将数学表达式的值
'''
from tensorflow.keras import models
from tensorflow.keras import layers

#创建sequential模型
network = models.Sequential()
#添加全连接层
network.add(layers.Dense(512,activation = 'relu',input_shape=(28*28,))) #激活函数为：relu ,隐藏层有512个结点
#添加输出层
network.add(layers.Dense(10,activation='softmax'))  #输出10种结果。 算概率，取概率最大的为1
#编译模型（优化项：rmsprop 均方根传播。 损失函数：交叉熵。评价标准：准确性)
network.compile(optimizer='rmsprop',loss= 'categorical_crossentropy',metrics=['accuracy'])
#打印模型结构（sequantial)
network.summary()

'''
4.数据归一化处理
    1.将二维数组转化成一维  
    2.将灰度图0-255转化为0-1
    3.to_categorical 将图片标签进行更改，转化成二进制  -- one hot encoding
'''

train_images = train_images.reshape((60000,28*28)) # 每行代表一个数
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical   #utils库包含许多简化代码编写的工具

print('before_change', train_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after_change',train_labels[0])

'''
5.执行训练过程
    1.epochs:完整循环次数。 代
    2.batch_size: 一批数据。 每次训练一批，需迭代多次完成一次完整循环
'''
network.fit(train_images,train_labels,epochs = 5,batch_size=150)

'''
6.测试数据输入，检验学习后的识别效果
    输出损失与精确度
'''
#verbose:整数，0：不输出，1：输出进度条 2：每个批次输出一行信息。
test_loss, test_acc = network.evaluate(test_images,test_labels,verbose=2)
print('test_loss',test_loss)
print('test_acc',test_acc)

'''
7.检测一张图片的效果
'''
#要展示图片，需要重新下载一下数据集，因为之前的数据结构已经改成1维了
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
number = test_images[1]
plt.imshow(number, cmap=plt.cm.gray_r)
plt.show()
#重新将数据改成1维，进行预测
test_images = test_images.reshape((10000, 28 * 28))
result = network.predict(test_images)  #高考环节，不用label了

for i in range(result[1].shape[0]): #result1的长度。一共有10位，代表着10种可能结果
    if(result[1][i] == 1):   # 找到概率值为1的类别的索引

        print('number is:',i)
        break
print(result[1])