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
        # 根据输入的训练数据更新结点链路的权重。
        # 将输入的数据转化成numpy支持的二维矩阵，  .T将矩阵转成784个输入点为一列
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

    # 测试过程
    def query(self, inputs):
        # 隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


# 设定框架
# 一张图片总共有28*28 = 784个数值，因此需要让网络的输入层具备784个输入节点
input_nodes = 784
hidden_nodes = 220
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 输入训练数据
training_file = open("dataset/mnist_train.csv", 'r')
training_list = training_file.readlines()  # 按行读取，本文件共100行
training_file.close()

# 设定整体训练的循环次数
epochs = 5
for e in range(epochs):
    for record in training_list:
        all_values = record.split(',')
        # asfarry:转变为浮点型  [1:] 是因为0位是标签，表示是什么数字 后面的784位是输入点数据
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  # 将input转化为0.1-1之间
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 测试：
test_file = open("dataset/mnist_test.csv")
test_list = test_file.readlines()
test_file.close()

scores = []
for record in test_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])  # 标签位
    print("该图片对应的数字为:", correct_number)
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    label = np.argmax(outputs)  # 找数组中的最大索引,也就是该数字对应的编号
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print('scores:', scores)

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)  # 得分平均值
