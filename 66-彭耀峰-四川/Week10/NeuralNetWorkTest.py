'''
从零实现神经网络
'''

import scipy.special
import numpy

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化权重矩阵
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        # 设置激活函数sigmoid()
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        # 把inputs_list和targets_list转换成numpy支持的二维矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号通过激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_input = numpy.dot(self.who, hidden_outputs)
        # 输出层信号经过激活函数得到最终结果
        final_output = self.activation_function(final_input)

        # 计算误差,用于更新网络节点之间的权重
        output_error = targets - final_output
        hidden_error = numpy.dot(self.who.T, output_error * final_output * (1 - final_output))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_error * final_output * (1 - final_output)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_error * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputnodes):
        # 根据输入计算并输出结果
        # 计算中间层从输入层接收到的信号量
        hidden_inumpyuts = numpy.dot(self.wih, inputnodes)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inumpyuts)
        # 计算输出层接收到的信号量
        final_inumpyuts = numpy.dot(self.who, hidden_outputs)
        # 计算输出层经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inumpyuts)
        print(final_outputs)
        return final_outputs


# 初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 输入训练数据
training_data_file = open("mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

scores = []
test_data_list = open('mnist_test.csv')
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为：", correct_number)
    # 预处理数字图片，归一化处理
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print("output result is :", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
    print(scores)

# 计算成功率
scores_array = numpy.asarray(scores)
print("成功率为：", scores_array.sum() / scores_array.size)
