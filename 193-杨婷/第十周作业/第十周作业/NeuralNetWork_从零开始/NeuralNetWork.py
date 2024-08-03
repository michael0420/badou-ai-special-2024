import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        '''
        0.0：这是正态分布的均值（mean），意味着生成的随机数将围绕0分布。
        pow(self.hnodes, -0.5)：这是正态分布的标准差（std）。
        这里使用了pow函数（等价于**操作符）来计算self.hnodes的负0.5次方。
        这种初始化方法通常被称为He初始化（对于ReLU激活函数）
        或Xavier/Glorot初始化（对于sigmoid或tanh激活函数，
        但此处以负0.5次方应用，可能更偏向于He初始化的变种）。
        目的是使得前向传播和反向传播时，各层的激活值和梯度方差保持一致，有助于避免梯度消失或爆炸问题。
        (self.hnodes,self.inodes)：这是输出数组的形状
        '''
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))

        self.who += self.lr * np.dot(output_errors*final_outputs*(1-final_outputs),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),
                                     np.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print('softmax result:', final_outputs)
        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 512
    output_nodes = 10
    learning_rate = 0.1

    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 执行训练
    training_data_file = open('dataset/mnist_train.csv', 'r')  # 只读
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 5
    for i in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs_list = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            targets_list = np.zeros(output_nodes) + 0.01
            targets_list[int(all_values[0])] = 0.99
            n.train(inputs_list, targets_list)

    # 执行推理
    score = []
    test_data_file = open('dataset/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        print('网络检测数字为：', label)

        if label == correct_number:
            score.append(1)
        else:
            score.append(0)
print(score)

score_array = np.asarray(score)
print('正确率为：', score_array.sum()/score_array.size)






