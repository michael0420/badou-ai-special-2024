import numpy as np
import scipy.special
import time

class network:
    def __init__(self, innodes, hidenodes, outnodes, lr):
        self.innodes = innodes
        self.hidenodes = hidenodes
        self.outnodes = outnodes
        self.lr = lr
        self.wih = np.random.rand(self.hidenodes, self.innodes) - 0.5  # 输入层到隐藏层的权重
        self.who = np.random.rand(self.outnodes, self.hidenodes) - 0.5  # 隐藏层到输出层的权重
        self.activationfunc = lambda x: scipy.special.expit(x)

    def predit(self, inputs):
        hideinput = np.dot(self.wih, inputs)
        hideoutput = self.activationfunc(hideinput)
        outinput = np.dot(self.who, hideoutput)
        outoutput = self.activationfunc(outinput)
        #  print(outoutput)
        return outoutput

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        # 正向传播
        hideinput = np.dot(self.wih, inputs)
        hideoutput = self.activationfunc(hideinput)
        outinput = np.dot(self.who, hideoutput)
        outoutput = self.activationfunc(outinput)

        outputerror = targets - outoutput
        hiddenerror = np.dot(self.who.T, outputerror*outoutput*(1-outoutput))
        self.who += self.lr * np.dot((outputerror * outoutput *(1 - outoutput)), hiddenerror.T)
        self.wih += self.lr * np.dot((hiddenerror * hideoutput * (1 - hideoutput)), inputs.T)

if __name__ == '__main__':
    mynetwork = network(28*28, 80, 10, 0.3)

    # 读数据
    data_file = open('dataset/mnist_train.csv')
    data_list = data_file.readlines() #  数据全部存进来
    data_file.close()
    print("训练图片有:", len(data_list))
    epochs = 7
    #  循环读取每张图
    for e in range(epochs):
        start_time = time.time()
        for index in range(len(data_list)):
            values = data_list[index].split(',')
            #  去掉标签 转成矩阵
            inputs = (np.asfarray(values[1:]))/255.0
            #  转成0.01 和 0.99
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            mynetwork.train(inputs, targets)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"训练一代时间为：{elapsed_time} 秒")

    data_test_file = open('dataset/mnist_test.csv')
    data_test_list = data_test_file.readlines()
    data_test_file.close()

    success = 0
    for index in range(len(data_test_list)):
        value = data_test_list[index].split(',')
        correct = int(value[0])
        print("正确数字为:", correct)


        inputs = (np.asfarray(value[1:]))/255.0
        outputs = mynetwork.predit(inputs)
        number = np.argmax(outputs)
        print("推理的数字为:", number)
        if number == correct:
            success += 1

    print("正确率为:", success/10.0)






