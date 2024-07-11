import numpy as np
import scipy.special

"""
1 定义初始化
2 初始化两个权重

"""

class NeuralNetWork:

    #初始化
    def __init__(self,input,hinput,oinput,lr):
        self.inodes = input   #输入数量
        self.hnodes = hinput  #中间层数量
        self.onodes = oinput  # 输出层数量
        self.lr = lr          #学习率
        #输入层到隐藏层权重矩阵
        self.wih = np.random.normal(0,pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        #中间层到输出层权重矩阵
        self.who = np.random.normal(0,pow(self.onodes, -0.5),(self.onodes,self.hnodes))
        #激活函数
        self.activation_function = lambda x:scipy.special.expit(x)


    #训练
    def train(self,inputs_list, targets_list):

        inputs_list = np.array(inputs_list,ndmin =2).T
        targets_list = np.array(targets_list,ndmin=2).T

        #隐藏层输入
        hidden_input = np.dot(self.wih ,inputs_list)
        # 经过激活函数
        hidden_output = self.activation_function(hidden_input)
        #计算输出层输入
        output_input = np.dot(self.who ,hidden_output)
        #经过激活函数
        output = self.activation_function(output_input)

        #计算误差
        output_errors = targets_list - output
        #计算隐藏层损失函数
        hidden_erros = np.dot(self.who.T, output_errors * output * (1-output))
        #反向
        self.who += self.lr * np.dot(output_errors * output * (1 - output), np.transpose(hidden_output))
        self.wih += self.lr * np.dot(hidden_erros * hidden_output * (1 - hidden_output) , inputs_list.T)

    def query(self, input):
        # 隐藏层输入
        hidden_input = np.dot(self.wih, input)
        # 经过激活函数
        hidden_output = self.activation_function(hidden_input)
        # 计算输出层输入
        output_input = np.dot(self.who, hidden_output)
        # 经过激活函数
        output = self.activation_function(output_input)
        print(output)
        return output


input = 784
hinput = 200
oinput = 10
lr = 0.1
n = NeuralNetWork(input, hinput, oinput, lr)

#读取训练数据
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练次数
epochs = 5
for i in range(epochs):
    for record in training_data_list:
        data = record.split(',')
        training_data = (np.asfarray(data[1:])) / 255 * 0.99 + 0.01
        target = np.zeros(oinput) + 0.01
        # 第一列的数据为图片实际值，然后根据索引写入0.99
        target[int(data[0])] = 0.99
        n.train(training_data, target)

#读取测试数据
test_data_file = open("dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
score = []
for record in test_data_list:
    data = record.split(',')
    test_data = (np.asfarray(data[1:])) / 255 * 0.99 + 0.01
    # 第一列的数据为图片实际值，然后根据索引写入0.99
    print("实际图片为",int(data[0]))
    output = n.query(test_data)
    target = np.argmax(output)
    print("测试得到结果图片为：", target)
    if (target == int(data[0])):
        score.append(1)
    else:
        score.append(0)

print("score: ", score)
scores_array = np.asarray(score)
print("准确率为：", scores_array.sum()/scores_array.size)