# 2.手动实现简单神经网络
import numpy
import scipy.special

class network:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learingrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lrate = learingrate

        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_funtion = lambda x: scipy.special.expit(x)

        pass

    # 训练函数
    def train(self, input_list, target_list):

        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # 开始正向传播
        # 计算输入层到隐藏层的数据
        hiddens = numpy.dot(self.wih,inputs)
        # 将隐藏层数据过激活函数得到隐藏层输出
        outhiddens = self.activation_funtion(hiddens)
        # 计算隐藏层输出到输出层的数据
        outputs = numpy.dot(self.who, outhiddens)
        # 将输出层过激活函数得到输出
        finalouts = self.activation_funtion(outputs)
        # 计算误差
        output_errors = targets - finalouts
        hiddens_errors = numpy.dot(self.who.T, output_errors * finalouts * (1 - finalouts))

        # 反向传播更新权重
        self.who += self.lrate * numpy.dot((output_errors * finalouts * (1 - finalouts)), numpy.transpose(outhiddens))

        self.wih += self.lrate * numpy.dot((hiddens_errors * outhiddens * (1 - outhiddens)), numpy.transpose(inputs))
        pass

    # 推理函数
    def query(self, test_inputs):
        # 根据输入数据计算并输出答案
        # 计算输入层到隐藏层的数据
        hiddens = numpy.dot(self.wih,test_inputs)
        # 将隐藏层数据过激活函数得到隐藏层输出
        outhiddens = self.activation_funtion(hiddens)
        # 计算隐藏层输出到输出层的数据
        outputs = numpy.dot(self.who, outhiddens)
        # 将输出层过激活函数得到输出
        finalouts = self.activation_funtion(outputs)
        print(finalouts)
        return finalouts


# 初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
inputnodes = 784
hiddennodes = 300
outputnodes = 10
learingrate = 0.1
network = network(inputnodes,hiddennodes,outputnodes,learingrate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
train_data=open('D:/jawu/homework_10/code/dataset/mnist_train.csv','r')
train_data_list=train_data.readlines()
train_data.close()

#加入epocs,设定网络的训练循环次数
epocs=5
for i in range(epocs):
    # 把数据依靠','区分，并分别读入
    for record in train_data_list:
        allva=record.split(",")
        inputs=(numpy.asfarray(allva[1:]))/255.0*0.99+0.01
        # 设置图片与数值的对应关系
        targets=numpy.zeros(outputnodes)+0.01
        targets[int(allva[0])]=0.99
        network.train(inputs,targets)

#读入测试数据
#open函数里的路径根据数据存储的路径来设定
test_data=open('D:/jawu/homework_10/code/dataset/mnist_test.csv','r')
test_data_list=test_data.readlines()
test_data.close()
sorces=[]
for record in test_data_list:
    all_values=record.split(",")
    corrent_number=int(all_values[0])
    print("该图片对应的数字为:", corrent_number)
    # 预处理数字图片
    inputs=(numpy.asfarray(all_values[1:]))/255*0.99+0.01
    outputs=network.query(inputs)
    lable=numpy.argmax(outputs)
    print("推理出来的图片数字是",lable)
    if lable==corrent_number:
        sorces.append(1)
    else:
        sorces.append(0)

print(sorces)

#计算图片判断的成功率
scores_array = numpy.asarray(sorces)
print("成功率 = ", scores_array.sum() / scores_array.size)




