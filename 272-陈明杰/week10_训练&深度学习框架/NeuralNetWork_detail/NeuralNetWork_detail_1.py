import scipy.special
import numpy as np


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 1、初始化输入层，隐藏层，输出层的结点数，以及学习率
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 2、初始化权重
        # self.wih（输入层到隐藏层的权重矩阵）将会是一个 hidden_nodes*input_nodes 的矩阵，其元
        # 素值在 [-0.5, 0.5) 范围内随机生成
        self.wih = np.random.rand(hidden_nodes, input_nodes) - 0.5
        # self.who（隐藏层到输出层的权重矩阵）将会是一个 output_nodes*hidden_nodes 的矩阵，其元
        # 素值在 [-0.5, 0.5) 范围内随机生成
        self.who = np.random.rand(output_nodes, hidden_nodes) - 0.5

        # 初始化激活函数
        # scipy.special.expit是sigmoid激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def Softmax(self,z):
        exp_z = np.exp(z - np.max(z))  # 减去max(z)是为了防止数值溢出
        return exp_z / np.sum(exp_z)  # 归一化，使得所有概率之和为1

    def train(self, inputs_list, targets_list):
        # 因为inputs_list和targets_list都是一维数组，需要把他变为列向量，即二维数组，格式要求
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        print(f'inputs.shape={inputs.shape}')
        print(f'targets.shape={targets.shape}')
        # 从输入层到隐藏层
        hidden_layer_input = np.dot(self.wih, inputs)
        # 通过激活函数
        hidden_layer_output = self.activation_function(hidden_layer_input)
        # 从隐藏层到输出层
        output_layer_input = np.dot(self.who, hidden_layer_output)
        # 通过激活函数
        # output_layer_output = self.activation_function(output_layer_input)
        output_layer_output = self.Softmax(output_layer_input)

        # 通过得到的输出，计算输出层中输出结果与预期结果的误差
        # 在训练结束之前targets一定是比output_layer_output小的
        # 所以output_error一定是负数，相当于：新权值=旧权值-学习率*梯度 中的负号
        output_error = targets - output_layer_output
        # print(f'output_error={output_error}')
        # print(f'targets={targets}')
        # print(f'output_layer_output={output_layer_output}')
        # 计算隐藏层的输出结果与预期结果的误差
        hidden_error = np.dot(self.who.T, output_layer_output * (1 - output_layer_output) * output_error)
        # 利用误差反向传播更新权值
        self.who = self.who + self.learning_rate * np.dot(output_error * output_layer_output * (1 - output_layer_output)
                                                          , np.transpose(hidden_layer_output))
        self.wih = self.wih + self.learning_rate * np.dot(hidden_error * hidden_layer_output * (1 - hidden_layer_output)
                                                          , np.transpose(inputs))

    def query(self, inputs):
        # 推理
        # 从输入层到隐藏层
        hidden_layer = np.dot(self.wih, inputs)
        # 过激活函数
        hidden_layer = self.activation_function(hidden_layer)
        # 从隐藏层到输出层
        output_layer = np.dot(self.who, hidden_layer)
        # 因为手写数字识别有10个类别，所以输出层应该过softmax激活函数得到概率分布
        # outputs = self.activation_function(output_layer)
        outputs = self.Softmax(output_layer)
        print(outputs)
        sum=outputs.sum()
        print(sum)
        return outputs


# 初始化构建神经网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据并对神经网络进行训练
train_file = open("dataset/mnist_train.csv")
train_data_lists = train_file.readlines()
train_file.close()
epochs = 20
for i in range(epochs):
    for input in train_data_lists:
        # 因为读取的是excel文件，所以同一个样本中的每一个元素是以‘,’隔开的，所以以
        # ‘,’为分隔符，用所有元素构成一个一维数组
        all_values = input.split(',')
        # image_array是一个一维数组，表示一张图片的所有像素点，因为在文件中，每一行
        # 是一个样本，即一张图片，而第一个元素表示这张图片的标签，即该图片表示的数字(正确答案),
        # 所以需要去除第一个元素
        image_array = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 正确答案
        correct_number = int(all_values[0])
        # 把标签标为one hot格式
        targets = np.zeros(10) + 0.01
        # 第一个元素是几就令对应的位置为one hot的最大值
        targets[correct_number] = 0.99
        # 对这张图片进行训练
        # print(f'targets={targets}')
        network.train(image_array, targets)

# 对测试数据进行推理，检查神经网络训练的成果
# 打开对应的文件
test_file = open("dataset/mnist_test.csv")
# 按行读取，data_lists有n个元素，每个元素都是一个一维数组，代表一个样本，在这里代表一张图片的所有像素点
test_data_lists = test_file.readlines()
test_file.close()
print(test_data_lists[0])
scores = []
for input in test_data_lists:
    # 因为读取的是excel文件，所以同一个样本中的每一个元素是以‘,’隔开的，所以以
    # ‘,’为分隔符，用所有元素构成一个一维数组
    all_values = input.split(',')
    # test_image是一个一维数组，表示一张图片的所有像素点，因为在文件中，每一行
    # 是一个样本，即一张图片，而第一个元素表示这张图片的标签，即该图片表示的数字(正确答案),
    # 所以需要去除第一个元素
    test_image = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    # all_values的第一个元素就是这张图片的正确答案答案
    correct_number = all_values[0]
    print('-------------------------')
    print(correct_number)
    # 推理，返回的outputs是一个概率数组，分别表示检测结果是每对应类别的概率
    outputs = network.query(test_image)
    index = np.argmax(outputs)
    print(index)
    print('-------------------------')
    # 如果概率最大的位置对应的下标==正确答案，那么说明推理正确
    if int(index) == int(correct_number):
        scores.append(1)
    else:
        scores.append(0)

print(scores)

# 计算图片判断的准确率
accuracy = np.sum(scores) / len(test_data_lists)
print(f'accuracy = {accuracy}')
