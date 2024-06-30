import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        设置初始化
        :param inputnodes: 输入层
        :param hiddennodes: 中间层
        :param outputnodes: 输出层
        :param learningrate: 学习率
        """
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate

        # 初始化权重矩阵,wih表示输入层中和中间层之间的权重矩阵，who表示中间层和输出层之间的权重矩阵
        self.wih = np.random.rand(self.hiddennodes, self.inputnodes) - 0.5
        self.who = np.random.rand(self.outputnodes, self.hiddennodes) - 0.5
        print(self.wih)
        print(self.who)

        # 设置激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """
        根据输入的训练数据更新节点链路权重
        :param inputs_list: 输入的训练数据
        :param targets_list:    训练数据对应的正确结果
        :return:
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算从输入层到中间层的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 进行激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算从中间层到输出层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))


        self.who += self.learningrate * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                               np.transpose(hidden_outputs))

        self.wih += self.learningrate * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                               np.transpose(inputs))
        pass

    def query(self, inputs):
        # 计算从输入层到中间层的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 进行激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算从中间层到输出层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    # 初始化网络
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.1
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 读取图片 csv中每一行代表一个图片的像素
    training_data_file = open("../../imgs/digit_shouxie/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # 设置网络的训练循环次数
    epochs = 5
    for e in range(epochs):
        # 把数据依靠','区分，并分别读入
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            # 设置图片与数值的对应关系
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    test_data_file = open("../../imgs/digit_shouxie/mnist_test.csv")
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
        # 找到数值最大的神经元对应的编号
        label = np.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    # 计算图片判断的成功率
    scores_array = np.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
