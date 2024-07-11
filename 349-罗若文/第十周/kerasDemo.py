from tensorflow.keras.datasets import mnist #导入数据集

#1.加载数据集(包括训练数据60000张，测试数据10000张)
(train_images,train_lables),(test_images,test_lables) = mnist.load_data()

#2.构建神经网络结构
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()#设置模型类型
#设置输入层
#layers.Dense(…):构造一个数据处理层,Dense全连接输入层
# 512:节点个数   activation：激活函数  input_shape:输入的数据形状
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) #input_shape 的值需要是一个包含单个元素的元组 (28*28,)，而不是 28*28

#设置输出层
#10个分类
network.add(layers.Dense(10,activation='softmax'))

#编译神经网络模型，指定优化器，损失函数(MSE/交叉熵)和评估指标(正确率)
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#3.将数据进行归一化处理，转换标签的形式(one-hot)
#二维转一维数组
train_images = train_images.reshape((60000,28*28))
test_images = test_images.reshape((10000,28*28))
#将像素值映射到0-1之间
# astype:修改数据类型
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32') / 255

#将标签数据转换为one-hot编码
from tensorflow.keras.utils import to_categorical
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)

#4.开始训练
network.fit(train_images,train_lables,batch_size=128,epochs=2)

#5.测试神经网络
#verbose=1表示输出带进度条的日志信息
#返回的第一个值是损失，第二个是正确率
test_loss,test_acc = network.evaluate(test_images,test_lables,verbose=1)
print("accuracy:",test_acc)
print("loss:",test_loss)

#6.测试单张图片
from PIL import Image
import numpy as np
image_path='six.jpg'
image = Image.open(image_path).convert('L')#读取图片并转换成灰度图
image = image.resize((28,28))
#将图片转换为numpy数组，并进行归一化
image_array = np.array(image)
image_array = image_array.astype('float32')/255

image_array = image_array.reshape((1,28*28))
#预测
res = network.predict(image_array)

#返回值最大的类的索引
prediction = np.argmax(res)
print("预测值为：",prediction)