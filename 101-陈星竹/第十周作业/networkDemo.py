import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learning_rate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodeds = outputnodes
        self.learn_rate = learning_rate

        #设置权重
        self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        self.who = np.random.rand(self.onodeds,self.hnodes)-0.5

        #设置激活函数sigmod
        self.activation_function = lambda x:scipy.special.expit(x)
    def train(self,inputs_list,targets_list):
        #前向
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        outputs_input = np.dot(self.who, hidden_outputs)
        outputs = self.activation_function(outputs_input)

        #误差
        output_error = targets - outputs #Etotal对a01求偏导，a01对Etotal的影响,偏导结果为 总误差-输出层输出
        a_z_efffect = outputs*(1-outputs) #a01对z01求偏导 z01对a01的影响,偏导结果为a01*(1-a01)
        '''
        使用 self.who.T 是为了将输出层的误差正确地反向传播到隐藏层。这一步的关键在于权重矩阵的转置，它确保了矩阵维度的正确性，并按照权重矩阵的连接关系将误差分配回隐藏层的每个神经元。
        '''
        hidden_error = np.dot(self.who.T,output_error*a_z_efffect)
        #反向更新
        #新权重 = 旧权重 + 学习率 x （误差对权重的偏导）
        self.who += self.learn_rate * np.dot(output_error*a_z_efffect,np.transpose(hidden_inputs)) #转置方向才能点乘
        self.wih += self.learn_rate * np.dot(hidden_error*(hidden_outputs*(1-hidden_outputs)),np.transpose(inputs)) #转置方向才能点乘


    def query(self,inputs_list):
        #转置是为了矩阵乘
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        outputs_input = np.dot(self.who,hidden_outputs)
        outputs = self.activation_function(outputs_input)
        print(outputs)
        return outputs

#初始化网络
input_nodes = 784
output_nodes = 10
hidden_nodes = 100
learn_rate = 0.1

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learn_rate)

#读入数据
train_data_file = open('NeuralNetWork_从零开始/dataset/mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

#迭代次数
epoch = 5

for e in range(epoch):
    #分割数据和标签，标签进行one-hot转换
    for record in train_data_list:
        all_values = record.split(',')
        #归一化处理输入数据
        inputs = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
        #one-hot转换
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99 #all_value[0]代表图片的标签为x，这一步是将target中索引为x的值改为0.99，其他值为0.01
        #训练
        n.train(inputs,targets)

#验证集验证
test_data_file = open('NeuralNetWork_从零开始/dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应数字：", correct_number)
    # 归一化处理输入数据
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    #推理
    target = n.query(inputs)
    res = np.argmax(target)
    print("该模型预测数字：",res)
    if res == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)