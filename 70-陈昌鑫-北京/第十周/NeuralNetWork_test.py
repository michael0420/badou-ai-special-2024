import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        #更新权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#图片28*28=784，所以需要784个输入
input_nodes = 784
#隐藏层，自定义
hidden_nodes = 150
#输出层，0-9，共10个数
output_nodes = 10
#学习率，自己设置
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#拿数据
training_data_file = open("dataset/mnist_test.csv", "r")
#按行读取，返回所有行的列表
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练循环次数
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        #数据按 ‘，’号分开
        all_values = record.split(',')
        # 归一且避免为0
        inputs = (numpy.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
        # 设置一个对应关系，最小值为0.01，10类
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0]是label，假如为7，则设第8个值为0.99
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为：", correct_number)
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    #找到最大值所在的索引
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是： ", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)
#将输入转换为浮点数类型的数组
scores_array = numpy.asfarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)