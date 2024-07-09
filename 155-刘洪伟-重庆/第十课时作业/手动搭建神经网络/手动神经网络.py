# _*_ coding: UTF-8 _*_
# @Time: 2024/6/27 18:42
# @Author: iris
# @Email: liuhw0225@126.com
import numpy
import numpy as np
import scipy.special


class NeuralNetWork(object):

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.input_nodes = inputNodes
        self.hidden_nodes = hiddenNodes
        self.output_nodes = outputNodes
        # 设置学习率
        self.learning_rate = learningRate

        """
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        """
        self.wih = (numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes)))
        self.who = (numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes)))
        '''
        设置激活函数，用sigmoid作为激活函数
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def fix(self, inputs_list, targets_list):
        """
            根据输入的训练数据更新节点链路权重
        :param inputs_list: 输入
        :param targets_list: 标签
        :return:
        """
        """
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        """
        input_array = numpy.array(inputs_list, ndmin=2).T
        target_array = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, input_array)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层经过激活函数后产生的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层经过激活函数后产生的信号量
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = target_array - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重得更新量，然后把更新加到原来链路权重上
        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                np.transpose(hidden_outputs))
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                np.transpose(input_array))

    def evaluate(self, input_array):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, input_array)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


if __name__ == '__main__':
    """
        初始化网络
        input_nodes: 输入节点个数
        hidden_nodes: 隐藏节点个数
        output_nodes: 输出节点个数
        learning_rate: 学习率
    """
    input_nodes = 28 * 28
    hidden_nodes = 1024
    output_nodes = 10
    learning_rate = 0.1

    network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    """准备训练数据"""
    train_files = open('datas/mnist_train.csv')
    train_images = train_files.readlines()
    train_files.close()

    # epochs 循环次数
    epochs = 10
    for i in range(epochs):
        for record in train_images:
            # 数据第一列为label标签
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            # 设置图片与数值的对应关系
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            network.fix(inputs, targets)

    """准备测试数据"""
    test_files = open("datas/mnist_test.csv")
    test_images = test_files.readlines()
    test_files.close()
    scores = []
    for record in test_images:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 预处理数字图片
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = network.evaluate(inputs)
        # 找到数值最大的神经元对应的编号
        label = numpy.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)

    # 计算图片判断的成功率
    scores_array = numpy.asarray(scores)
    print("performance = ", scores_array.sum() / scores_array.size)
