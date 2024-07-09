'''
使用numpy写一个简单的神经网络
'''
import numpy as np
import scipy.special

class NeuralNetwork():

    def __init__(self, inputnodes, hidnodes, outputnodes, learningrate):
        self.nodes1 = inputnodes
        self.nodes2 = hidnodes
        self.nodes3 = outputnodes
        self.rate = learningrate
        self.w1 = np.random.normal(0.0, pow(self.nodes2, -0.5), (self.nodes2, self.nodes1))
        self.w2 = np.random.normal(0.0, pow(self.nodes3, -0.5), (self.nodes3, self.nodes2))
        self.activation_func = lambda x: scipy.special.expit(x)

    def training(self, train_data, train_labels):
       # print(self.w2,'***************************')
        inputs = np.array(train_data, ndmin=2).T
        labels = np.array(train_labels, ndmin=2).T

        # 求每层函数的经过加权、激活后的值
        zh = np.dot(self.w1, inputs)
        ah = self.activation_func(zh)       #512*128
        zo = np.dot(self.w2, ah)
        ao = self.activation_func(zo)       #10*128

        # 计算损失
        out_error = labels - ao             # 10*128
        hid_error = np.dot(self.w2.T, out_error*ao*(1 - ao))    # 512*128

        # 反向更新权重
        self.w2 += self.rate * np.dot(out_error*ao*(1-ao), ah.T)
        self.w1 += self.rate * np.dot(hid_error*ah*(1-ah), inputs.T)
       # print(self.w2,'***************************')

    def test(self, inputs, labels):
        scores = []
        leng = inputs.shape[0]
        for i in range(leng):
            zh = np.dot(self.w1, inputs[i])
            ah = self.activation_func(zh)
            zo = np.dot(self.w2, ah)
            ao = self.activation_func(zo)
            print(ao)
            label = np.argmax(ao)
            t = np.argmax(labels[i])
            print(f'第{i}条数据：\n'
                  f'预测结果是：{label}\n'
                  f'实际结果是：{t}')
            if label == t:
                scores.append(1)
            else:
                scores.append(0)

        scores_array = np.asarray(scores)
        print(scores)
        print("perfermance = ", scores_array.sum() / scores_array.size)

        return ao


def readdata(f):

        # 读取数据
        train_data_file = open(f, 'r')
        train_data_list = np.asarray(train_data_file.readlines())
        train_data_file.close()
        train_data = []
        train_labels = []
        for i in train_data_list:
            i = i.split(',')
            train_data.append(i[1:])
            train_labels.append(int(i[0]))

        # 对数据进行处理
        train_data = (np.asfarray(train_data) / 255.0) * 0.99 + 0.01
        one_hot = np.zeros((10, 10)) + 0.01
        np.fill_diagonal(one_hot, 0.99)
        one_hot = one_hot[train_labels]

        return train_data, one_hot

if __name__ == '__main__':

    # 初始化模型
    network = NeuralNetwork(28*28, 512, 10, 0.01)

    # 获取数据
    train_file = 'dataset/mnist_train.csv'
    test_file = 'dataset/mnist_test.csv'
    train_data, train_labels = readdata(train_file)
    test_data, test_labels = readdata(test_file)

    # 训练
    epoch = 1000
    bantch_size = 1
    cout = int(len(train_data) / bantch_size + 0.5)
    for i in range(epoch):
        for j in range(cout):
            inputs = train_data[bantch_size * i: bantch_size * (i+1)]
            labels = train_labels[bantch_size * i: bantch_size * (i+1)]
            network.training(inputs, labels)

    network.test(test_data, test_labels)









