import scipy.special
import numpy


def glorot_init(shape):
    limit = numpy.sqrt(6 / (shape[0] + shape[1]))
    return numpy.random.uniform(-limit, limit, shape)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化网络，设置节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 设置学习率
        self.learning_rate = learning_rate

        '''
        初始化权重矩阵：
        weight_input_hidden表示输入层和中间层的权重矩阵
        weight_hidden_output表示隐藏层和输出层的权重矩阵
        '''
        # self.weight_input_hidden = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.weight_hidden_output = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        # self.weight_input_hidden = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # self.weight_hidden_output = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.weight_input_hidden = glorot_init((self.hidden_nodes, self.input_nodes))
        self.weight_hidden_output = glorot_init((self.output_nodes, self.hidden_nodes))

        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_error = numpy.dot(self.weight_hidden_output.T, output_errors * final_outputs * (1 - final_outputs))

        self.weight_hidden_output += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                                    numpy.transpose(hidden_outputs))
        self.weight_input_hidden += self.learning_rate * numpy.dot((hidden_error * hidden_outputs * (1 - hidden_outputs)),
                                                                   numpy.transpose(inputs))

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_output = self.activation_function(final_inputs)
        print(final_output)
        return final_output


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.1

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读入数据
    training_data_file = open("dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 1
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01

            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    test_data_file = open('dataset/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)

    scores_array = numpy.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
