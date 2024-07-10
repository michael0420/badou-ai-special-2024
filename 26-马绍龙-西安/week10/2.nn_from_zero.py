import numpy as np
import scipy.special


class nn_manual:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        self.activation = lambda x: scipy.special.expit(x)  # 用sigmoid作为激活函数

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs

    def train(self, input_list, answers_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(answers_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        # 反向传播
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs))


# 构建网络参数，进行主方法调用
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = nn_manual(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 路径可动态变化
training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10
# 加入epochs,设定网络的训练循环次数
for e in range(epochs):
    for record in trainning_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 对targets标签进行one hot编码
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 测试训练后的模型
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让模型判断图片对应的数字,推理
    outputs = n.query(inputs)
    # print(outputs)
    # 找到数值最大的神经元对应的 索引值
    label = np.argmax(outputs)
    print("output reslut is(推理值) : ", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的正确率
scores_array = np.asarray(scores)
print("perfermance(正确率) = ", scores_array.sum() / scores_array.size)
print("=============================")

# 使用其他图片进行推理
import cv2

path = "dataset/my_own_image.png"
img = cv2.imread(path, 0)
pic = (np.asfarray(img).reshape(784)) / 255.0 * 0.99 + 0.01
result = n.query(pic)
print(result)  # 结果的准确率很低，是因为训练数据过于简单，适配不了自己写的数据
print(path, "推理出的数字为:", np.argmax(result))
