[1]
"""
step1:正向传播
inputs_list:输入的训练数据;
targets_list:训练数据对应的正确结果。
"""
import numpy as np


def train(self, inputs_list, target_list):
    '''
    把inputs_list, targets_list转换成numpy支持的二维矩阵
    .T表示做矩阵的转置
    '''
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(target_list, ndim=2).T
    # 计算信号经过输入层后产生的信号量
    hidden_inputs = np.dot(self.wih, inputs)
    # 中间层神经元对输入的信号做激活函数后得到输出信号
    hidden_outputs = self.activation_function(hidden_inputs)
    # 输出层接收来自中间层的信号量
    final_inputs = np.dot(self.who, hidden_outputs)
    # 输出层对信号量进行激活函数后得到最终输出信号
    final_outputs = self.activation_function(final_inputs)

[2]
'''
上面代码根据输入数据计算出结果后，我们先要获得计算误差.
误差就是用正确结果减去网络的计算结果。
在代码中对应的就是(targets - final_outputs).
'''


def train(self, inputs_list, targets_list):
    # 根据输入的训练数据更新节点链路权重
    """
    把inputs_list, targets_list转换成numpy支持的二维矩阵
    .T表示做矩阵的转置
    """
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T
    # 计算信号经过输入层后产生的信号量
    hidden_inputs = np.dot(self.wih, inputs)
    # 中间层神经元对输入的信号做激活函数后得到输出信号
    hidden_outputs = self.activation_function(hidden_inputs)
    # 输出层接收来自中间层的信号量
    final_inputs = np.dot(self.who, hidden_outputs)
    # 输出层对信号量进行激活函数后得到最终输出信号
    final_outputs = self.activation_function(final_inputs)

    # 计算误差(套公式)
    output_errors = targets - final_outputs  # Etotal/ao1的偏导
    hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))  # Eo1/ah1的偏导
    # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
    self.who += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
    self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))

    pass

[3]
'''
使用实际数据来训练我们的神经网络
'''
data_file = open('dataset/mnist_train.csv')
data_list = data_file.readlines()
data_file.close()
print(len(data_list))  # 一行代表一张图片，查看一共几张图片
print(data_list[0])
'''
这里我们可以利用画图.py将输入绘制出来
'''

[4]
'''
从绘制的结果看，数据代表的确实是一个黑白图片的手写数字。
数据读取完毕后，我们再对数据格式做些调整，以便输入到神经网络中进行分析。
我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
'''
scaled_inputs = image_array / 255 * 0.99 + 0.01




