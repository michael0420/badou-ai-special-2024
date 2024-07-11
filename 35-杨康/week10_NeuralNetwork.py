import numpy as np
import scipy.special

class NeuralNetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.n_input = inputnodes
        self.n_hidden = hiddennodes
        self.n_output = outputnodes
        self.lr = learningrate
        #用sigmoid作为激活函数
        self.active_function = lambda x:scipy.special.expit(x)
        #设置输入层到隐藏层和隐藏层到输出层的权重为符合正态分布的随机数
        self.wih = np.random.normal(0,pow(self.n_hidden,-0.5),(self.n_hidden,self.n_input))
        self.who = np.random.normal(0,pow(self.n_output,-0.5),(self.n_output,self.n_hidden))

    def train(self,inputslist,targetslist):
        inputs = np.array(inputslist,ndmin=2).T
        targets = np.array(targetslist,ndmin=2).T
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.active_function(hidden_in)
        output_in = np.dot(self.who, hidden_out)
        output_out = self.active_function(output_in)
        out_error = targets - output_out
        hidden_error = np.dot(self.who.T,out_error*output_out*(1-output_out))
        self.who += self.lr*np.dot(out_error*output_out*(1-output_out),np.transpose(hidden_out))
        self.wih += self.lr*np.dot(hidden_error*hidden_out*(1-hidden_out),np.transpose(inputs))
        pass

    def query(self,inputs):
        hidden_in = np.dot(self.wih,inputs)
        hidden_out = self.active_function(hidden_in)
        output_in = np.dot(self.who,hidden_out)
        output_out = self.active_function(output_in)
        print(output_out)
        return output_out

#由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读取训练数据
train_data_file = open("dataset/mnist_train.csv",'r')
train_data_file_list = train_data_file.readlines()
train_data_file.close()
epoch = 5
for i in range(epoch):
    for record in train_data_file_list:
        record = record.split(',')
        inputs = np.asfarray(record[1:])/255.0*0.99+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(record[0])] = 0.99
        n.train(inputs,targets)
#读取测试数据
test_data_file = open("dataset/mnist_test.csv",'r')
test_data_file_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_file_list:
    record = record.split(',')
    print('图片对应的数字是：', record[0])
    inputs = np.asfarray(record[1:])/255.0*0.99+0.01
    outputs = n.query(inputs)
    num = np.argmax(outputs)
    print('判断的数字是：',num)
    if num == int(record[0]):
        scores.append(1)
    else:
        scores.append(0)
print(scores)
#计算图片判断的成功率
scores_array = np.array(scores)
print("判断的成功率： ", scores_array.sum() / scores_array.size)