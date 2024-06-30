import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rete):
        # 初始化输入层节点个数，隐藏层节点个数，输出层节点个数，学习率
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 创建输入层到隐藏层的权重矩阵，初始化隐藏层到输出层的权重矩阵，并初始化为随机值
        self.wih = np.random.rand(hidden_nodes, input_nodes) - 0.5
        self.who = np.random.rand(output_nodes, hidden_nodes) - 0.5

        # 初始化激活函数，scipy.special.expit(x)表示sigmid函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练
    def train(self, inputs_x_list, inputs_y_list):
        inputs_x = np.array(inputs_x_list, ndmin=2).T
        inputs_y = np.array(inputs_y_list, ndmin=2).T
        # 输入层到隐藏层
        hidden_layers_inputs = np.dot(self.wih, inputs_x)
        # 隐藏层的输入过激活函数得到隐藏层的输出
        hidden_layers_outputs = self.activation_function(hidden_layers_inputs)
        # 隐藏层到输出层
        output_layers_inputs = np.dot(self.who, hidden_layers_outputs)
        # 输出层的输入过激活函数得到输出层的输出
        output_layers_outputs = self.activation_function(output_layers_inputs)

        # 计算输出层的误差，即预期输出结果与实际输出结果的差，反过来减是为了有个负号
        output_layers_errors = inputs_y - output_layers_outputs
        # 计算隐藏层的误差，即预期的隐藏层输出结果与实际的隐藏层输出结果的差
        hidden_layers_errors = np.dot(self.who.T
                                      , output_layers_outputs * (1 - output_layers_outputs) * output_layers_errors)

        # 更新隐藏层到输出层的权重矩阵
        self.who = self.who + self.learning_rate * np.dot(
            output_layers_errors * output_layers_outputs * (1 - output_layers_outputs)
            , np.transpose(hidden_layers_outputs))

        # 更新输入层到隐藏层的权重矩阵
        self.wih = self.wih + self.learning_rate * np.dot(
            hidden_layers_errors * hidden_layers_outputs * (1 - hidden_layers_outputs)
            , np.transpose(inputs_x))

    # 推理
    def query(self, inputs_x):
        # 输入层到隐藏层
        hidden_layers_inputs = np.dot(self.wih, inputs_x)
        # 隐藏层的输入过激活函数得到隐藏层的输出
        hidden_layers_outputs = self.activation_function(hidden_layers_inputs)
        # 隐藏层到输出层
        output_layers_inputs = np.dot(self.who, hidden_layers_outputs)
        # 输出层的输入过激活函数得到输出层的输出
        output_layers_outputs = self.activation_function(output_layers_inputs)
        # 曾经这里的打印写成了print(output_layers_inputs)，导致结果展示出错，这太坑了
        print(output_layers_outputs)
        return output_layers_outputs


# 创建神经网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取数据进行训练
train_data_file = open("dataset/mnist_train.csv")
train_datas = train_data_file.readlines()
train_data_file.close()
eporch = 20
for i in range(eporch):
    for inputs in train_datas:
        # 以','分隔开并格式化为一个一维数组
        one_image = inputs.split(',')
        # 一张图片的数据是一个一维矩阵，第一个元素是该图片的正确答案，归一化
        image_data = np.asfarray(one_image[1:]) / 255.0 * 0.99 + 0.01
        # 正确答案
        image_correct_number = int(one_image[0])
        # 正确答案变成one hot格式
        target = np.zeros(10) + 0.01
        target[image_correct_number] = 0.99
        # 训练
        network.train(image_data, target)

# 推理
test_data_file = open("dataset/mnist_test.csv")
test_datas = test_data_file.readlines()
test_data_file.close()
scores = []
for inputs in test_datas:
    one_image = inputs.split(',')
    image_data = np.asfarray(one_image[1:]) / 255.0 * 0.99 + 0.01
    image_correct_number = int(one_image[0])
    print('-------------------------')
    print(image_correct_number)
    res = network.query(image_data)
    query_res = np.argmax(res)
    print(query_res)
    print('-------------------------')
    if query_res == image_correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)
print(f'correct_accuracy = {np.sum(scores) / len(scores)}')
