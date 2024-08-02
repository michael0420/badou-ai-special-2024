[1]
'''
这套代码无法运行，是写代码的思路，一点一点往里面添加内容
先写出代码框架
'''

import numpy as np


class NerualNetWork:
    def __init__(self):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        pass

    def train(self):
        # 根据输入的训练数据更新节点链路权重
        pass

    def query(self):
        # 根据输入数据计算并输出答案
        pass

[2]
'''
我们先完成初始化函数，我们需要在这里设置输入层，中间层和输出层的节点数，这样就能决定网络的形状和大小。
当然我们不能把这些设置都写死，而是根据输入参数来动态设置网络的形态。
由此我们把初始化函数修正如下：
'''
class NerualNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes  # 输入层节点个数
        self.hnodes = hiddennodes  # 隐藏层节点个数
        self.onodes = outputnodes  # 输出层节点个数
        self.lr = learningrate  # 学习率
        pass

    def train(self):
        # 根据输入的训练数据更新节点链路权重
        pass

    def query(self):
        # 根据输入数据计算并输出答案
        pass

[3]
'''
此处举例说明：
如此我们就可以初始化一个3层网络，输入层，中间层和输出层都有3个节点
'''
inputnodes = 3  # 一旦发生变化只需在这里更改参数
hiddennodes = 3
outputnodes = 3
learningrate = 0.3

n = NerualNetWork(inputnodes, hiddennodes, outputnodes, learningrate)

[4]
'''
初始化权重矩阵。
由于权重不一定都是正的，它完全可以是负数，因此我们在初始化时，把所有权重初始化为-0.5到0.5之间
'''


class NerualNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        import numpy as np
        self.inodes = inputnodes  # 输入层节点个数
        self.hnodes = hiddennodes  # 隐藏层节点个数
        self.onodes = outputnodes  # 输出层节点个数
        self.lr = learningrate  # 学习率
        pass
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5  # random.rand生成0-1之间随机数
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        pass

[5]
'''
矩阵乘得到隐藏层节点输入
**推理和正向训练过程一样，先写推理可以复制到训练里
'''


def query(self, inputs):
    hidden_inputs = np.dot(self.wih, inputs)
    pass

[6]
'''
设置激活函数（sigmod）
'''
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.inodes) - 0.5

        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义.
        我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


'''
由此我们就可以分别调用激活函数计算中间层的输出信号，以及输出层经过激活函数后形成的输出信号，
'''
def query(self, inputs):
    hidden_inputs = np.dot(self.wih, inputs)  # 计算中间层从输入层接收到的信号量
    hidden_outputs = self.activation_function(hidden_inputs)  # 计算中间层经过激活函数后形成的输出信号量
    final_inputs = np.dot(self.who, hidden_outputs)  # 计算最外层接收到的信号量
    final_outputs = self.activation_function(final_inputs)  # 计算最外层神经元经过激活函数后输出的信号量
    print(final_outputs)
    return final_outputs
