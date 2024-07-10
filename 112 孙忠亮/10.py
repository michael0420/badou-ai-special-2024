from tensorflow.keras.datasets import mnist #导入数据集

(train_images,train_lables),(test_images,test_lables) = mnist.load_data()

#2.构建神经网络结构
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()#设置模型类型
#设置输入层
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
#设置输出层
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#3.将数据进行归一化处理，转换标签的形式(one-hot)
train_images = train_images.reshape((60000,28*28))
test_images = test_images.reshape((10000,28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32') / 255
#将标签数据转换为one-hot编码
from tensorflow.keras.utils import to_categorical
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)
#4.开始训练
network.fit(train_images,train_lables,batch_size=128,epochs=2)
#5.测试神经网络
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

import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learning_rate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodeds = outputnodes
        self.learn_rate = learning_rate

        #设置权重
        self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        self.who = np.random.rand(self.onodeds,self.hnodes)-0.5

        #设置激活函数sigmod
        self.activation_function = lambda x:scipy.special.expit(x)
    def train(self,inputs_list,targets_list):
        #前向
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        outputs_input = np.dot(self.who, hidden_outputs)
        outputs = self.activation_function(outputs_input)

        #误差
        output_error = targets - outputs #Etotal对a01求偏导，a01对Etotal的影响,偏导结果为 总误差-输出层输出
        a_z_efffect = outputs*(1-outputs) #a01对z01求偏导 z01对a01的影响,偏导结果为a01*(1-a01)
        hidden_error = np.dot(self.who.T,output_error*a_z_efffect)
        self.who += self.learn_rate * np.dot(output_error*a_z_efffect,np.transpose(hidden_inputs)) #转置方向才能点乘
        self.wih += self.learn_rate * np.dot(hidden_error*(hidden_outputs*(1-hidden_outputs)),np.transpose(inputs)) #转置方向才能点乘


    def query(self,inputs_list):
        #转置是为了矩阵乘
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        outputs_input = np.dot(self.who,hidden_outputs)
        outputs = self.activation_function(outputs_input)
        print(outputs)
        return outputs

#初始化网络
input_nodes = 784
output_nodes = 10
hidden_nodes = 100
learn_rate = 0.1

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learn_rate)

#读入数据
train_data_file = open('NeuralNetWork_从零开始/dataset/mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

#迭代次数
epoch = 5

for e in range(epoch):
    #分割数据和标签，标签进行one-hot转换
    for record in train_data_list:
        all_values = record.split(',')
        #归一化处理输入数据
        inputs = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
        #one-hot转换
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99 #all_value[0]代表图片的标签为x，这一步是将target中索引为x的值改为0.99，其他值为0.01
        #训练
        n.train(inputs,targets)

#验证集验证
test_data_file = open('NeuralNetWork_从零开始/dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应数字：", correct_number)
    # 归一化处理输入数据
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    #推理
    target = n.query(inputs)
    res = np.argmax(target)
    print("该模型预测数字：",res)
    if res == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
x_test = np.linspace(-0.5,0.5,50)[:,np.newaxis]
noise_test = np.random.normal(0,0.02,x_test.shape)
y_test = np.square(x_test) + noise_test

#定义数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
wih = tf.Variable(tf.random.normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
#计算
wih_x_plus_b = tf.matmul(x,wih)+b1
#激活函数输出
l1 = tf.nn.tanh(wih_x_plus_b) #[1,10]

#定义神经网络输层
who = tf.Variable(tf.random.normal([10,1]))
# tf.matmul(l1,who) => [1,1]
b2 = tf.Variable(tf.zeros([1,1]))
who_x_plus_b = tf.matmul(l1,who)+b2
l2 = tf.nn.tanh(who_x_plus_b)
# 定义损失函数
loss = tf.reduce_mean(tf.square(y-l2))
#定义反向传播函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction = sess.run(l2,feed_dict={x:x_test})

    #画图
    plt.figure()
    plt.scatter(x_test,y_test)
    # plt.scatter(x_data, y_data)
    plt.plot(x_test,prediction,'r-',lw=5)
    plt.show()