import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate

        # 初始化权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.hiddennodes, self.inputnodes))
        # 初始化权重矩阵
        self.who = np.random.normal(0.0, pow(self.outputnodes, -0.5), (self.outputnodes, self.hiddennodes))

        # self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, x, y):
        x = np.array(x, ndmin=2).T
        y = np.array(y, ndmin=2).T
        hidden_inputs = np.dot(self.wih, x)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        outputs_err = y - final_outputs
        # sigmoid激活函数的倒数f'(x) = f(x) * (1 - f(x))
        hidden_err = np.dot(self.who.T, outputs_err * final_outputs * (1 - final_outputs))

        self.who += self.learningrate * np.dot((outputs_err * final_outputs * (1 - final_outputs)), hidden_outputs.T)
        self.wih += self.learningrate * np.dot((hidden_err * hidden_outputs * (1 - hidden_outputs)), x.T)

    def query(self, x):
        # 输入层与隐藏层之间的连接
        hidden_inputs = np.dot(self.wih, x)
        hidden_activations = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_activations)
        final_activations = self.activation_function(final_inputs)
        print(f"final_activations:{final_activations}")
        return final_activations


input_nodes = 784
hidden_nodes = 128
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练
# 数据集的shape（100,785）
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 5
for _ in range(epochs):
    for i in range(len(training_data_list)):
        all_data = training_data_list[i].split(',')
        x = (np.asfarray(all_data[1:])) / 255 * 0.99 + 0.01
        y = np.zeros(output_nodes) + 0.01
        # 第一个为标签
        y[int(all_data[0])] = 0.99
        n.train(x, y)

# 测试
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到概率最大的下标
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print("成功率：", scores_array.sum() / scores_array.size)
