import scipy.special
import numpy as np

class NeuralNetWork:
    # 初始化 输入层，隐藏层，输出层 节点个数，学习率激活函数等属性
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        # 初始化权重
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 激活函数(给属性指定默认值)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 输入层到隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层 到输入层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新量加到原来的 链路  权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(inputs))
        pass

    def query(self,inputs):
        '对输入数据进行推理，输入层 ，隐藏层，输出层'
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# 初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取数据 open函数里的路径 根据数据存储的 路径来设定
training_data_file = open("mnist_train.csv",'r')
training_data_list= training_data_file.readlines() # 按照行读取，所以读取出来时100个长度的元素
training_data_file.close()

# 加入epoch设置网络：
epochs = 5
for e in range(epochs):
    # 把数据按照“，”隔开，这些读进来的全部是字符串的形式
    for record in training_data_list:
        all_values =record.split(',')
        # 取除了第一位的所有数，也就是图片的像素值，总共784个
        inputs = (np.asfarray(all_values[1:]))/255
        # 设置标签，图片与标签对应上
        targets = np.zeros(output_nodes)
        targets[int(all_values[0])] = 1
        # 把图片和标签进行训练
        n.train(inputs, targets)

test_data_file = open('mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores =[]
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("这张照片的正确数字是：",correct_number)
    inputs = (np.asfarray(all_values[1:])) / 255
    outputs = n.query(inputs)
    label =np.argmax(outputs)
    print("网络预测这张图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)










