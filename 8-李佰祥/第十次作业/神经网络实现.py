import random

import numpy
import numpy as np
import scipy.special


class NNetWork:
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )
        self.who = (numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )

        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        #计算隐藏层输入
        hidden_inputs = np.dot(self.wih,inputs)
        #计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层的输入
        final_inputs = np.dot(self.who,hidden_outputs)
        #计算最终输出
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))

        self.who += self.lr * np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rates = 0.1
model =NNetWork(input_nodes,hidden_nodes,output_nodes,learning_rates)

train_data_file = open('./NeuralNetWork_从零开始/dataset/mnist_train.csv')
train_data_list = train_data_file.readlines()
train_data_file.close()
#print(train_data_list)
epochs=5
for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])) /255.0 *0.99 +0.1
        #设置图片与数值之间的对应关系
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] =0.99
        #print(inputs.shape,targets.shape)
        model.train(inputs,targets)

#读取测试集
test_data_file = open('./NeuralNetWork_从零开始/dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores =[]

for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    inputs =  (numpy.asfarray(all_values[1:])) /255 *0.99 +0.1
    outputs = model.query(inputs)
    label = np.argmax(outputs)
    print("网络认为图片的数字为：",label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

score_array = numpy.asarray(scores)
print(score_array.sum()/score_array.size)





