import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        #初始化各层和学习率
        self.in_nodes = input_nodes
        self.hid_nodes = hidden_nodes
        self.output_node = output_nodes
        self.learn_rate = learn_rate

        #初始化权重矩阵
        self.wih = numpy.random.rand(self.hid_nodes, self.in_nodes) - 0.5
        self.who = numpy.random.rand(self.output_node, self.hid_nodes) - 0.5

        #设置激活函数sigmoid
        self.activation = lambda x:scipy.special.expit(x)

        pass

    #推理过程
    def query(self, inputs):
        hid_value = numpy.dot(self.wih, inputs)
        hid_output = self.activation(hid_value)

        out_value = numpy.dot(self.who, hid_output)
        out_output = self.activation(out_value)
        print(out_output)
        return out_output

    #训练过程
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hid_value = numpy.dot(self.wih, inputs)
        hid_output = self.activation(hid_value)

        out_value = numpy.dot(self.who, hid_output)
        out_output = self.activation(out_value)

        #计算误差
        output_errors = targets - out_output
        hid_errors = numpy.dot(self.who.T, output_errors * out_output * (1 - out_output))
        self.who += self.learn_rate * numpy.dot(output_errors * out_output * (1 - out_output), numpy.transpose(hid_output))
        self.wih += self.learn_rate * numpy.dot(hid_errors * hid_output * (1 - hid_output), numpy.transpose(inputs))

        pass


#初始化
input_nodes = 28*28
hide_nodes = 200
out_nodes = 10
learn_rate = 0.1
n = NeuralNetWork(input_nodes, hide_nodes, out_nodes, learn_rate)

#读取数据
train_data = open("mnist_train.csv")
train_data_list = train_data.readlines()
train_data.close()

#设定epoch
epochs = 5
for i in range(epochs):
    #已经通过读取行设置了batch
    for data in train_data_list:
        all_values = data.split(',')
        #确保范围在0。01到1之间
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(out_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

#验证训练结果
test_data = open("mnist_test.csv")
test_data_list = test_data.readlines()
test_data.close()
scores = []

for data in test_data_list:
    all_values = data.split(',')
    cur_number = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if label == cur_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

sum1 = numpy.asarray(scores)
print("rate = ", sum1.sum() / sum1.size)