import numpy
import scipy.special
class NeuralNetWork:
    def __init__(self, inodes, hnodes, onodes, lr):
        #设置输入层、中间层、输出层节点数，设置学习率
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.lr = lr
        #初始化输入层到中间层的权重wih，中间层到输出层的权重who
        # wih --> hnodes行，inodes列
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        #初始化激活函数
        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义.
        我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x : scipy.special.expit(x)


    def query(self, inputs):
        #计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算中间层经过激活函数后，输出的信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层从中间层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算输出层经过激活函数后，输出的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后，输出的信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层从中间层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算输出层经过激活函数后，输出的信号量
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        # 更新who,whi
        outputs_errors = targets - final_outputs
        self.who += self.lr * numpy.dot(outputs_errors * final_outputs * (1 - final_outputs), numpy.transpose(hidden_outputs))
        hidden_errors = numpy.dot(self.who.T, outputs_errors*final_outputs*(1-final_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))






