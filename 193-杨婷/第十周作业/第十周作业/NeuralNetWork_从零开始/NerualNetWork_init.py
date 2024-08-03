'''
可以用于执行，没有实际意义，验证代码是否有错误
'''

import numpy as np
import scipy.special

class NeruralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = np.random.rand(hiddennodes, inputnodes) - 0.5
        self.who = np.random.rand(outputnodes, hiddennodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self):
        pass

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs

'''
我们尝试传入一些数据，让神经网络输出结果试试.
程序当前运行结果并没有太大意义，但是至少表明，我们到目前为止写下的代码没有太大问题，
'''
inputnodes = 3
hiddennodes = 3
outputnodes = 3
learningrate = 0.3

n = NeruralNetWork(inputnodes, hiddennodes, outputnodes, learningrate)
n.query([1.0, 0.5, -1.5])
