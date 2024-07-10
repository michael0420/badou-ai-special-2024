import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,lr):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inode = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = lr
        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inode)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        # 每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        #lambda x:为匿名函数，等同于建立一个只有一行的def函数，returnscipy.special.expit(x)
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self,input_list,target_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = np.array(input_list,ndmin =2).T
        targets = np.array(target_list,ndmin =2).T
        # 正向传播
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs*(1-final_outputs))

        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who +=self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)),np.transpose(hidden_outputs))
        self.wih +=self.lr * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),np.transpose(inputs))


    def query(self,inputs):
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

#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_val = record.split(',')
        inputs = (np.asfarray(all_val[1:]))/255.0 * 0.99 + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_val[0])] = 0.99
        network.train(inputs,targets)
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_vals = record.split(',')
    number = int(all_vals[0])
    print("该图片对应的数字为:",number)
    # 预处理数字图片
    inputs = (np.asfarray(all_vals[1:]))/255.0 * 0.99 + 0.01
    outputs = network.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ",scores_array.sum()/scores_array.size)