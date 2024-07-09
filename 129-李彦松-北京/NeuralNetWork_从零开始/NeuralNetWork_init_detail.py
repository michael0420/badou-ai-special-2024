[1]
'''
我们先给出如下代码框架：
'''
class  NeuralNetWork:
    def __init__(self):
        #初始化网络，设置输入层，中间层，和输出层节点数
        pass
    def  train(self):
        #根据输入的训练数据更新节点链路权重
        pass
    def  query(self):
        #根据输入数据计算并输出答案
        pass
        
[2]
'''
我们先完成初始化函数，我们需要在这里设置输入层，中间层和输出层的节点数，这样就能决定网络的形状和大小。
当然我们不能把这些设置都写死，而是根据输入参数来动态设置网络的形态。
由此我们把初始化函数修正如下：
'''
class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #设置学习率
        self.lr = learningrate

        pass
    def  train(self):
        #根据输入的训练数据更新节点链路权重
        pass
    def  query(self):
        #根据输入数据计算并输出答案
        pass

[3]
'''
此处举例说明：
如此我们就可以初始化一个3层网络，输入层，中间层和输出层都有3个节点
'''
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

[4]
'''
初始化权重矩阵。
由于权重不一定都是正的，它完全可以是负数，因此我们在初始化时，把所有权重初始化为-0.5到0.5之间
'''
def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        pass

[5]
'''
接着我们先看query函数的实现，它接收输入数据，通过神经网络的层层计算后，在输出层输出最终结果。
输入数据要依次经过输入层，中间层，和输出层，并且在每层的节点中还得执行激活函数以便形成对下一层节点的输出信号。
我们知道可以通过矩阵运算把这一系列复杂的运算流程给统一起来。
'''
import numpy
def  query(self, inputs):
        #根据输入数据计算并输出答案
        hidden_inputs = numpy.dot(self.wih, inputs)
        pass

[6]
'''
hidden是个一维向量，每个元素对应着中间层某个节点从上一层神经元传过来后的信号量总和.
于是每个节点就得执行激活函数，得到的结果将作为信号输出到下一层.
sigmod函数在Python中可以直接调用，我们要做的就是准备好参数。
我们先把这个函数在初始化函数中设定好，
'''
import scipy.special

class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义.
        我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x:scipy.special.expit(x)

        pass

'''
由此我们就可以分别调用激活函数计算中间层的输出信号，以及输出层经过激活函数后形成的输出信号，
'''
def  query(self, inputs):
        #根据输入数据计算并输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
        




