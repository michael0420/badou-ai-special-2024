import cv2
import numpy as np


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.activation = lambda x: 1 / (1 + np.exp(-x))

        # 定义随即权重
        self.wih = np.random.rand(hidden_nodes, input_nodes) - 0.5
        self.who = np.random.rand(output_nodes, hidden_nodes) - 0.5
        # print(self.wih.shape)
        # print(self.who.shape)
        pass

    # def Softmax(self, x):
    #     # 获取x的维度
    #     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减去每一行的最大值以防止溢出
    #     s = e_x / e_x.sum(axis=1, keepdims=True)  # 计算Softmax值
    #     return s
    def Softmax(self,z):
        exp_z = np.exp(z - np.max(z))  # 减去max(z)是为了防止数值溢出
        return exp_z / np.sum(exp_z)  # 归一化，使得所有概率之和为1

    def train(self, data, label):
        data = data.reshape(self.input_nodes, -1)
        label = label.reshape(self.output_nodes, -1)
        # print(data.shape)
        # print(label.shape)
        hidden_input = np.dot(self.wih, data)
        hidden_output = self.activation(hidden_input)
        output_input = np.dot(self.who, hidden_output)
        output_output = self.activation(output_input)
        # 输出层误差
        output_error = output_output - label
        # 隐藏层误差
        hidden_output_error = np.dot(self.who.T, output_output * (1 - output_output) * output_error)
        # 更新权重
        # 转置？
        self.who = self.who - self.learning_rate * np.dot(output_error * output_output * (1 - output_output),
                                                          hidden_output.T)
        self.wih = self.wih - self.learning_rate * np.dot(hidden_output_error * hidden_output * (1 - hidden_output),
                                                          data.T)

        pass

    def query(self, data):
        data = data.reshape(self.input_nodes, -1)
        hidden_input = np.dot(self.wih, data)
        hidden_output = self.activation(hidden_input)
        output_input = np.dot(self.who, hidden_output)
        pridiction = self.Softmax(output_input)
        return pridiction


# 初始化参数
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
net = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取文件进行训练
with open('dataset/mnist_train.csv', 'r') as f:
    train_data_lines = f.readlines()
epochs = 20
for i in range(epochs):
    for train_data in train_data_lines:
        list = train_data.split(',')
        list = np.asfarray(list)
        label = int(list[0])
        data = list[1:]/255*0.99+0.01
        one_hot_to_label = np.zeros(10)
        one_hot_to_label[label] = 0.99
        net.train(data, one_hot_to_label)

# 读取文件进行推理
with open('dataset/mnist_test.csv', 'r') as f:
    test_data_lines = f.readlines()
arr = []
count = 0
for test_data in test_data_lines:
    list = test_data.split(',')
    list = np.asfarray(list)
    label = int(list[0])
    data = list[1:]
    predition = net.query(data)
    print(predition)
    index = np.argmax(predition)
    if index == label:
        arr.append('1')
        count += 1
    else:
        arr.append('0')
    print('-------------------------------')
    print(f'label={label}')
    print(f'prediction={index}')
    print('-------------------------------')
print(arr)
print(f'accuracy={count / len(test_data_lines)}')
