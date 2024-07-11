import numpy
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络输入层、隐藏层、输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # 初始化权重矩阵
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        # 初始化激活函数,用sigmoid作为激活函数
        self.activation = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, target_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        output_inputs = numpy.dot(self.who, hidden_outputs)
        output_outputs = self.activation(output_inputs)
        # 更新权重
        output_errors = targets - output_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * output_outputs * (1 - output_outputs))
        self.who += self.lr * numpy.dot(output_errors * output_outputs * (1 - output_outputs),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        output_inputs = numpy.dot(self.who, hidden_outputs)
        output_outputs = self.activation(output_inputs)
        # print(output_outputs)
        return output_outputs


# 28*28 = 784
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
with open('dataset/mnist_train.csv', 'r') as trainfile:
    train_data_list = trainfile.readlines()
trainImages = []
trainLables = []
for record in train_data_list:
    all_values = record.split(',')
    trainImages.append(all_values[1:])
    trainLables.append(all_values[0])
# 数据归一化
trainImages = numpy.asfarray(trainImages) / 255.0 * 0.999 + 0.001
# 标记进行onehot编码
targets = numpy.zeros((len(trainLables), output_nodes)) + 0.001
for i in range(len(trainLables)):
    value = int(trainLables[i])
    targets[i][value] = 0.999
trainLables = targets
# 网络训练
epochs = 5
for i in range(epochs):
    for j in range(len(trainImages)):
        n.train(trainImages[j], trainLables[j])

# 读取数据
with open('dataset/mnist_test.csv', 'r') as test_data_file:
    test_data_list = test_data_file.readlines()
testImages = []
testLables = []
for record in test_data_list:
    all_values = record.split(',')
    testImages.append(all_values[1:])
    testLables.append(all_values[0])
# 数据归一化
testImages = numpy.asfarray(testImages) / 255.0 * 0.999 + 0.001
# 标记进行onehot编码
targets = numpy.zeros((len(testLables), output_nodes)) + 0.001
for i in range(len(testLables)):
    targets[i, int(testLables[i])] = 0.999
# testLables = targets
# 网络测试
outputs_list = []
score = []
testLength = len(testLables)
for i in range(testLength):
    outputs = n.query(testImages[i])
    print(numpy.argmax(outputs))
    if numpy.argmax(outputs) == int(testLables[i]):
        score.append(1)
    else:
        score.append(0)
    outputs_list.append(outputs)
# print([numpy.argmax(outputs_list[i]) for i in range(len(outputs_list))])
print('测试样本数：{}，正确率：{}'.format(testLength, sum(score) / testLength))
