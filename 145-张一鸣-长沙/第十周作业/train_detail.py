# coding = utf-8

'''
        手搓训练过程
'''


import numpy as np
import scipy.special

class train_detail:

    def __init__(self, input_nodes, output_nodes, hidden_nodes, learning_rate):
        # 初始化输入层、隐藏层、输出层
        self.input = input_nodes
        self.output = output_nodes
        self.hidden = hidden_nodes
        # 初始化学习率
        self.stu = learning_rate
        # 初始化权重，random.rand随机0-1之间的数，-0.5使初始值有正有负
        # self.w_ih = np.random.rand(self.hidden, self.input) - 0.5
        # self.w_ho = np.random.rand(self.output, self.hidden) - 0.5
        self.w_ih = (np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input)))
        self.w_ho = (np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden)))
        # 初始化激活函数，scipy.special.expit(x) 即 sigmod 函数
        self.activation = lambda x: scipy.special.expit(x)


    def train(self, inputs_list, targets_list):
        # 训练
        # 数据的前置处理，将输入的list转换为矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 与推理过程相同
        h_in = np.dot(self.w_ih, inputs)
        h_out = self.activation(h_in)

        o_in = np.dot(self.w_ho, h_out)
        o_out = self.activation(o_in)

        # 计算误差
        o_error = targets - o_out
        h_error = np.dot(self.w_ho.T, o_error * o_out * (1-o_out))

        # 计算新权重并更新，+=是因为损失梯度下降量自带方向
        self.w_ho += self.stu * np.dot((o_error * o_out * (1 - o_out)), np.transpose(h_out))
        self.w_ih += self.stu * np.dot((h_error * h_out * (1 - h_out)), np.transpose(inputs))


    def predict(self, inputs):
        # 推理（也是正向传播），将kx+b和激活过程代码化
        h_in = np.dot(self.w_ih, inputs)
        h_out = self.activation(h_in)

        o_in = np.dot(self.w_ho, h_out)
        o_out = self.activation(o_in)
        print('输出结果：', o_out)
        return o_out


if __name__ == '__main__':

    # 设置神经网络参数
    input_nodes = 28*28
    output_nodes = 10
    hidden_nodes = 100
    learning_rate = 0.5
    epochs = 8
    model = train_detail(input_nodes, output_nodes, hidden_nodes, learning_rate)

    # 读取训练数据
    mnist_data_train = open(r'dataset/mnist_train.csv')
    data_list_train = mnist_data_train.readlines()
    mnist_data_train.close()

    # 根据epoch次数进行迭代训练
    for k in range(epochs):
        for i in data_list_train:
            split_data_train = i.split(',')
            # 特别的数据标准化
            inputs = (np.asfarray(split_data_train[1:])) / 255.0 * 0.99 + 0.01
            # 设置图片和数值的对应关系
            targets = np.zeros(output_nodes) + 0.01
            targets[int(split_data_train[0])] = 0.99
            model.train(inputs, targets)


    # 读取测试数据
    mnist_data_test = open(r'dataset/mnist_test.csv')
    data_list_test = mnist_data_test.readlines()
    mnist_data_test.close()
    print('数据量：', len(data_list_test))
    print(data_list_test[0])

    # # 处理数据
    # split_data_test = data_list_test[0].split(',')
    # # 第一列为label，故训练数据从索引1开始
    # data_array = np.asfarray(split_data_test[1:]).reshape((28, 28))
    #
    # # 设置输出元素表示为0.99和0.01
    # onodes = 10
    # targets = np.zeros(onodes) + 0.01
    # targets[int(split_data_test[0])] = 0.99
    # print(targets)

    # 将测试集输入模型查看效果
    scores = []
    for j in data_list_test:
        values = j.split(',')
        labels = int(values[0])
        print('图片中的数字为：', labels)
        # 标准化处理防止数值失真
        inputs = (np.asfarray(values[1:])) / 255.0 * 0.99 + 0.01
        # 对测试集进行推理
        outputs = model.predict(inputs)
        # 找出数值最大的神经元对应编号
        result = np.argmax(outputs)
        print('推理结果为：', result)
        # 将结果放入列表，后续计算准确率
        if result == labels:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)

    # 计算准确率
    scores_array = np.asarray(scores)
    print('准确度为：', scores_array.sum() / scores_array.size)
