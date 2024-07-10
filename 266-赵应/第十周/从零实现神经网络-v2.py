import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, input_node, hidden_node, output_node, lr=0.01):
        #  初始化参数
        self.inode = input_node
        self.hnode = hidden_node
        self.onode = output_node
        self.lr = lr

        # 初始化权重矩阵，wih权重形状（hnode，inode），who权重形状（onode，hnode）
        # 权重形状如此设置目的是为了适配训练数据的形状，后期需要做矩阵乘法
        # 权值初始化算法：需要保证一部分为正数、一部分为负数，这样才能达到权值选择的效果，不然全是正向选择、或者负向选择，会导致模型产生错误
        # self.wih = np.random.rand(self.hnode, self.inode)
        self.wih = np.random.normal(0.0, pow(self.hnode, -0.5), (self.hnode, self.inode))
        # self.who = np.random.rand(self.onode, self.hnode)
        self.who = np.random.normal(0.0, pow(self.onode, -0.5), (self.onode, self.hnode))

        #  初始化激活函数, 默认选用sigmod。实际上可以作为参数传入
        self.activation = lambda x: scipy.special.expit(x)

    def fit(self, train_data, train_label):
        # 依次将样本数据喂给网络
        for i in range(train_data.shape[0]):
            data_input = np.array(train_data[i], ndmin=2).T
            data_label = np.array(train_label[i], ndmin=2).T
            # 正向传播过程
            # 数据由输入层经过隐藏层
            tmp = np.dot(self.wih, data_input)
            hidden_layer_output = self.activation(tmp)
            # 数据由隐藏层经过输出层
            output_layer_output = self.activation(np.dot(self.who, hidden_layer_output))

            #  计算误差并根据误差更新权值矩阵, 反向传播过程
            output_error = data_label - output_layer_output
            hidden_error = np.matmul(self.who.T, output_error * output_layer_output * (1 - output_layer_output))

            self.who += self.lr * np.matmul(output_error * output_layer_output * (1 - output_layer_output),
                                            np.transpose(hidden_layer_output))

            self.wih += self.lr * np.matmul(hidden_error * hidden_layer_output * (1 - hidden_layer_output),
                                            np.transpose(data_input))

    def evaluate(self, test_data, test_label):
        scores = []
        for i in range(test_data.shape[0]):
            # 数据由输入层经过隐藏层
            hidden_layer_output = self.activation(np.dot(self.wih, test_data[i]))
            # 数据由隐藏层经过输出层
            output_layer_output = self.activation(np.dot(self.who, hidden_layer_output))
            print(output_layer_output)
            evaluate_label = np.argmax(output_layer_output)
            print("网络检测到的数字是： ", evaluate_label)
            really_label = np.argmax(test_label[i])
            print("实际数字是：", really_label)
            if really_label == evaluate_label:
                scores.append(1)
            else:
                scores.append(0)
        print(scores)
        print("perfermance: ", np.sum(scores) / len(scores))


def load_dataset(file_path):
    with open(file_path) as file:
        source_data = file.readlines()
        sample_len = len(source_data)
        data_array = np.zeros((sample_len, 784))
        data_label = np.zeros((sample_len, 10)) + 0.001
        for i in range(sample_len):
            img_value = source_data[i].replace("\n", "").split(",")
            #  数据归一化，将数据缩放到（0.001 ~ 1）
            img_array = (np.asfarray(img_value[1:])) / 255.0 * 0.99 + 0.01
            data_array[i] = data_array[i] + img_array
            data_label[i][int(img_value[0])] = 0.99
    return data_array, data_label


if __name__ == '__main__':
    train_data, train_label = load_dataset("dataset/mnist_train.csv")
    test_data, test_label = load_dataset("dataset/mnist_test.csv")
    # 由于输入图片形状为28x28 因此输入层节点数为28*28
    input_node = 784
    # 隐藏层节点数可以随意设置
    hidden_node = 100
    # 在此做的是0~9十个手写数字的识别，因此输出层节点为10，分别代表0~9数组的概率
    output_node = 10
    # 训练轮数
    epoch = 100
    net = NeuralNetWork(input_node, hidden_node, output_node, 0.1)
    for i in range(epoch):
        net.fit(train_data, train_label)
    net.evaluate(test_data, test_label)
