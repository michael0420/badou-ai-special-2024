import numpy as np
import scipy.special
import cv2

class NetWork:
    def __init__(self,inputdim,hiddendim,outputdim,learningrate):
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.hiddendim = hiddendim
        self.learningrate = learningrate
        # 权重初始化 分为两部分，一个是输入到隐藏层的权重矩阵，一个是中间层和输出层链路权重形成的矩阵
        self.intohiddenw = (np.random.normal(0,pow(self.hiddendim,-0.5),(self.hiddendim,self.inputdim)))
        self.hiddentooutw = (np.random.normal(0,pow(self.outputdim,-0.5),(self.outputdim,self.hiddendim)))
        #实现sogimoid 激活函数
        self.activation = lambda x:scipy.special.expit(x)

    def train(self,inputs_list,target_list):
        # 根据输入的训练数据更新结点链路权重
        print("原来input得形状：",inputs_list.shape)
        inputs = np.array(inputs_list,ndmin=2).T
        print(inputs.shape)
        targets =np.array(target_list,ndmin=2).T
        hidden_inputs = np.dot(self.intohiddenw,inputs)
        hidden_outputs = self.activation(hidden_inputs)
        out_res = np.dot(self.hiddentooutw,hidden_outputs)
        out_res =self.activation(out_res)

        #反向传播
        output_errors = targets - out_res
        #求出梯度  根据公式 偏导E / 偏导W = -（traget - output）*sigmoid()*(1-sigmoid())*o

        # 隐藏层的误差
        hidden_errors = np.dot(self.hiddentooutw.T,output_errors*out_res*(1-out_res))
        self.intohiddenw += self.learningrate*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))
        self.hiddentooutw +=self.learningrate*np.dot(output_errors*out_res*(1-out_res),np.transpose(hidden_outputs))



    def query(self,inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.intohiddenw,inputs)
        hidden_outputs = self.activation(hidden_inputs)
        out_res = np.dot(self.hiddentooutw,hidden_outputs)
        out_res = self.activation(out_res)
        print(out_res)
        return out_res


# 由于输入的是28*28的图片，所以输入

input_nodes = 28*28
hidden_nodes = 200
output_nodes = 10
learning_rate =0.1
n =NetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)



train_data_file = open('dataset/mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

epochs = 5
for e in range(epochs):
    for record in train_data_list:
        all_data = record.split(',')
        inputs = (np.asfarray(all_data[1:])) / 255.0 *0.99 +0.01
        tragets = np.zeros(output_nodes)+0.01
        tragets[int(all_data[0])]=0.99
        print(inputs.shape)
        n.train(inputs,tragets)

test_data_file = open('dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()
socre=[]
for record in test_data_list:
    all_data = record.split(',')
    corect_num = int(all_data[0])
    print('正确的数字是： ',corect_num)
    inputs = (np.asfarray(all_data[1:])) / 255. *0.99 +0.01
    outputs = n.query(inputs)
    labels = np.argmax(outputs)
    print('网络认为的数字是： ',labels)
    if labels==corect_num:
        socre.append(1)
    else:
        socre.append(0)




scores_array = np.asarray(socre)
print("perfermance = ", scores_array.sum() / scores_array.size)













