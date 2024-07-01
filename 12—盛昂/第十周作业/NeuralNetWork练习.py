#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.special



# 一、神经网络的类的建立
class NeuralNetWork:
    #初始化参数
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes =inputnodes
        self.hnodes =hiddennodes
        self.onodes =outputnodes
        #学习率
        self.lr =learningrate
        #权重
        self.wih =np.random.rand(self.hnodes,self.inodes) -0.5
        self.who =np.random.rand(self.onodes,self.hnodes) -0.5
        #激活函数
        self.activation_function =lambda x:scipy.special.expit(x)
        pass
    #根据训练数据，更新线路节点的权重
    def train(self,input_list,target_list):
        inputs =np.array(input_list,ndmin=2).T
        targets =np.array(target_list,ndmin =2).T
        #计算输入信号至中间隐藏层节点
        hidden_inputs =np.dot(self.wih,inputs)
        hidden_outputs =self.activation_function(hidden_inputs)
        #计算隐藏层节点至输出节点
        final_inputs =np.dot(self.who,hidden_outputs)
        #输出层激活函数后得到的输出值
        final_outputs =self.activation_function(final_inputs)
        
        #计算误差
        output_errors =targets-final_outputs
        hidden_errors =np.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))
        self.who +=self.lr *np.dot((output_errors *final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih +=self.lr *np.dot((hidden_errors *hidden_inputs*(1-hidden_inputs)),np.transpose(inputs))
        pass
    #模型结构（再建立一遍，冗余）
    def query(self,inputs):
        hidden_inputs =np.dot(self.wih,inputs)
        hidden_outputs =self.activation_function(hidden_inputs)
        final_inputs =np.dot(self.who,hidden_outputs)
        final_outputs =self.activation_function(final_inputs)
        
        print(final_outputs)
        return final_outputs

    
# 二、实战演练*********************************  ***************************
# 初始化网络
input_nodes =784
hidden_nodes =200
output_nodes =10
learning_rate =0.1
n =NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# 读入训练数据
training_data_file =open("dataset/mnist_train.csv")
training_data_list =training_data_file.readlines()
training_data_file.close()

# 加入epochs，设定网络的训练循环次数
epochs =5
for e in range(epochs):
    for record in training_data_list:
        all_value =record.split(',')
        inputs =(np.asfarray(all_value[1:],))/255+0.1
        #设置图片与标签的关系
        targets =np.zeros(output_nodes)+0.01
        targets[int(all_value[0])] =0.99
        n.train(inputs,targets)
# 检测数据
test_data_file =open("dataset/mnist_train.csv")
test_data_list =test_data_file.readlines()
test_data_file.close()
scores=[]

# for record in test_data_list:
#     all_values =record.split(',')
#     correct_number =int(all_values[0])
#     print("该图片对应的数字为：", correct_number)
    
#     #预处理数字图片
#     inputs =(np.asfarray(all_values[1:]))/255*0.99+0.01
#     outputs =n.query(inputs)
#     label =np.argmax(outputs)
#     print("网络认为图片的数字是：",label)
#     if label ==correct_number:
#         scores.append(1)
#     else:
#         scores.append(0)
# print(scores)

# # 计算图片判断的成功率
# scores_array =np.asarray(scores)
# print("perfermance = ",scores_array.sum()/scores_array.size)
        

        





# In[ ]:





# In[ ]:




