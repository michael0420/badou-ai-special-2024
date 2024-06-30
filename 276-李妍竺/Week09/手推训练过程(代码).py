import numpy as np



import scipy.special  # special库，包含基本数学函数，特殊函数，以及numpy中的所有函数

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数,学习率
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        '''
        随机初始化权重矩阵 wih:输入与隐藏， who：隐藏与输出。 （0-1随机或高斯随机）
        生成权重，-0.5 是为了生成-0.5到0.5的值，因为权重可以是负数
        采用normal -0.5是用来调整标准差大小的，当节点很多的时候，需要方差小一些，所以取负
        '''
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))  # pow:^
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # 定义激活函数。 lambda:一种轻量级的函数定义方式  匿名的
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):

        epochs = 40000
        for e in range(epochs):
            # 根据输入的训练数据更新结点链路的权重。
            # 将输入的数据转化成numpy支持的二维矩阵，  .T将矩阵转成输入点为一列
            inputs = np.array(inputs_list, ndmin=2).T  # ndmin:定义数组的最小维度
            targets = np.array(targets_list, ndmin=2).T

            '''
            前向传播：输入层——>隐藏层——>输出层
            z:激活函数之前(input), a:激活函数之后(output)
            '''
            # 隐藏层
            hidden_inputs = np.dot(self.wih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            # 输出层
            final_inputs = np.dot(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            '''
            计算误差，进行反向传播
            output_error: target-a_o
            delta_output: -(target-a_o)*a_o*(1-a_o)    计算时，省去-，方便计算更新部分
            hidden_error: sum(delta_output*who)
            delta_hidden: hidden_error*a_h*(1-a_h)

            更新权重：
            who_new = who-lr*delta_output*ah
            wih_new = whi-lr*delta_hidden*input    
            '''
            output_errors = targets - final_outputs
            hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

            delta_output = output_errors * final_outputs * (1 - final_outputs)
            delta_hidden = hidden_errors * hidden_outputs * (1 - hidden_outputs)
            # 更新权重
            self.who += self.lr * np.dot(delta_output, np.transpose(hidden_outputs))
            self.wih += self.lr * np.dot(delta_hidden, np.transpose(inputs))



            if e % 1000 == 0:
                E_o1 = 0.5 * (expected_output[0] - final_outputs[0]) ** 2
                E_o2 = 0.5 * (expected_output[1] - final_outputs[1]) ** 2
                E_o3 = 0.5 * (expected_output[2] - final_outputs[2]) ** 2
                E_total = E_o1 + E_o2 + E_o3
                print(f'迭代 {e}, 总误差: {E_total}')


        print(f'训练后的最终输出: {final_outputs.flatten()}')
        print(f'最终总误差: {E_total}')


# 输入训练数据

input_nodes = 3
hidden_nodes = 6
output_nodes = 3
learning_rate = 0.1


ini_inputs = np.array([0.05,0.10,0.15])
expected_output = np.array([0.01,0.3,0.69])


n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.train(ini_inputs, expected_output)



