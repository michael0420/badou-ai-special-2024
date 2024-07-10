import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inodes, hnodes, onodes, learnning_rate):
        self.inputnodes_num = inodes
        self.hiddennodes_num = hnodes
        self.outputnodes_num = onodes
        self.r = learnning_rate
        self.wih = np.random.rand(self.hiddennodes_num, self.inputnodes_num)
        self.who = np.random.rand(self.outputnodes_num, self.hiddennodes_num)


    def query(self, input_data):
        hidden_input = np.dot(self.wih, input_data)
        hidden_output = scipy.special.expit(hidden_input)
        output_input = np.dot(self.who, hidden_output)
        output_output = scipy.special.expit(output_input)
        print(output_output)
        return output_output


    def trainning(self, input_data, target):
        hidden_input = np.dot(self.wih, input_data)
        hidden_output = scipy.special.expit(hidden_input)
        output_input = np.dot(self.who, hidden_output)
        output_output = scipy.special.expit(output_input)

        loss_output = (target - output_output)
        loss_hidden = np.dot(self.who.T, loss_output * output_output * (1 - output_output))
        self.who += self.r * np.dot((loss_output*output_output*(1 - output_output)), np.transpose(hidden_output))
        self.wih += self.r * np.dot((loss_hidden*hidden_output*(1 - hidden_output)), np.transpose(input_data))
        pass


inodes = 784
hnodes = 200
onodes = 10
r = 0.1
n = NeuralNetWork(inodes, hnodes, onodes, r)

trainning_data_file = open('dataset/mnist_train.csv', mode='r')
trainning_data_list = trainning_data_file.readlines()
trainning_data_file.close()
query_data_file = open('dataset/mnist_test.csv', mode='r')
query_data_list = query_data_file.readlines()
query_data_file.close()

for epoch in range(5):
    for row in trainning_data_list:
        row_list = row.split(',')
        target = np.zeros(onodes) + 0.001
        target[int(row_list[0])] = 0.99
        target = np.array(target, ndmin=2).T
        input_data = (np.array(row_list[1:], dtype=np.float32, ndmin=2)/255.0 * 0.99 + 0.01).T
        n.trainning(input_data, target)

for row_1 in query_data_list:
    row_1_list = row_1.split(',')
    input_test = (np.array(row_1_list[1:], dtype=np.float32, ndmin=2)/255.0 * 0.99 + 0.01).T
    outputs_test = n.query(input_test)
    sort = np.argsort(outputs_test.flatten())
    result = sort[9]
    print(f'神经网络判断此手写数字为:{result}')
    print(f'此手写数字实际为：{row_1_list[0]}')




