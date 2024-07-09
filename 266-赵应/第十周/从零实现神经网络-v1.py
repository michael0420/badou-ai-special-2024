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
        self.wih = np.random.rand(self.hnode, self.inode)
        self.who = np.random.rand(self.onode, self.hnode)

        #  初始化激活函数, 默认选用sigmod。实际上可以作为参数传入
        self.activation = lambda x: scipy.special.expit(x)

    def fit(self):
        pass

    def evaluate(self, train_data):
        # 数据由输入层经过隐藏层
        hidden_layer_output = self.activation(np.dot(self.wih, train_data))
        # 数据由隐藏层经过输出层
        output_layer_output = self.activation(np.dot(self.who, hidden_layer_output))
        return output_layer_output


if __name__ == '__main__':
    net = NeuralNetWork(4, 5, 4, 0.1)
    result = net.evaluate([1.0, 0.5, -1.5, 1.3])
    print(result)
