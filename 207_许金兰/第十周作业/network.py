"""
@author: 207-xujinlan
使用numpy 从零开始实现神经网络训练推理过程
"""

import numpy as np

class NetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        """
        初始化参数
        :param input_nodes: 输入层节点数
        :param hidden_nodes: 隐藏层节点数
        :param output_nodes: 输出层节点数
        :param lr: 学习率
        """
        self.lr = lr
        # 随机初始化模型参数,-0.5加快收敛速度
        self.wih = np.random.normal(0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
        self.who = np.random.normal(0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
        self.activation_func = lambda x: 1 / (1 + np.exp(-x))   # 激活函数

    def train(self, input_data, label):
        """
        模型训练
        :param input_data: 输入数据，训练数据
        :param label: 数据标签
        :return: 更新模型参数
        """
        input_data = np.array(input_data, ndmin=2).T
        label = np.array(label, ndmin=2).T
        hidden_input = np.dot(self.wih, input_data)
        hidden_output = self.activation_func(hidden_input)
        out_input = np.dot(self.who, hidden_output)
        out_output = self.activation_func(out_input)
        out_erros = label - out_output
        hidden_errors = np.dot(self.who.T, out_erros * out_output * (1 - out_output))
        self.who += self.lr * np.dot((out_erros * out_output * (1 - out_output)),
                                     np.transpose(hidden_output))
        self.wih += self.lr * np.dot((hidden_errors * hidden_output * (1 - hidden_output)),
                                     np.transpose(input_data))

    def predict(self, input_data):
        """
        模型预测
        :param input_data: 输入数据
        :return: 返回预测结果
        """
        hidden_input = np.dot(self.wih, input_data)
        hidden_output = self.activation_func(hidden_input)
        out_input = np.dot(self.who, hidden_output)
        out_output = self.activation_func(out_input)
        return out_output


# 读取训练数据
train_data_file = open('dataset/mnist_train.csv')
train_data = train_data_file.readlines()
train_data_file.close()

epoch = 5
input_nodes = 28 * 28
hidden_nodes = 200
output_nodes = 10
lr = 0.1
# 模型实例化
nw = NetWork(input_nodes, hidden_nodes, output_nodes, lr)
# 模型训练
for i in range(epoch):
    for line in train_data:
        all_values = line.split(',')
        input_data = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
        label = np.zeros(output_nodes)
        label[int(all_values[0])] = 0.99
        nw.train(input_data, label)

# 模型测试和评分
test_data_file = open('dataset/mnist_test.csv')
test_data = test_data_file.readlines()
test_data_file.close()
score = []
for line in test_data:
    test_values = line.split(',')
    print('true number is:', test_values[0])
    input_data = np.asfarray(test_values[1:]) / 255 * 0.99 + 0.01
    result = np.argmax(nw.predict(input_data))
    print('model predict number is :', result)
    if result == int(test_values[0]):
        score.append(1)
    else:
        score.append(0)
print('model score is :', np.sum(score))
