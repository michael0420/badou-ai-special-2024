'''
采用手写搭建神经网络框架，梳理结构形成
'''
import numpy as np
import matplotlib.pyplot as plt

[12]  # 补充训练循环步骤
class NeuralNetwork:
    def __init__(self, nodes_in, nodes_hide, nodes_out, learning_rate):  # 方便后续计算调用，设置各层节点数及学习率
        self.nodes_in = nodes_in
        self.nodes_hide = nodes_hide
        self.nodes_out = nodes_out
        self.lr = learning_rate
        self.wih = np.random.random(size=[self.nodes_hide, self.nodes_in]) - 0.5  # wih表示权重矩阵in->hide, [0, 1) - 0. 5
        self.who = np.random.random(size=[self.nodes_out, self.nodes_hide]) - 0.5  # who表示权重矩阵hide->out
        import scipy.special  # 增加sigmoid激活函数初始化。
        self.activation = lambda x: scipy.special.expit(x)
        # 类C语言宏定义.调用self.activation(x)时，编译器会把其转换为spicy.special_expit(x)
        # expit函数，也称logistic sigmoid函数，定义为expit（x）= 1 /（1 + exp（-x））。 是logit函数的反函数
        pass
    # 示例：nn = NeuralNetwork(3, 5, 3, 0.1)代表输入节点3，中间层节点5，输出层节点3， 学习率0.1
    def train(self, list_in, list_out):
        # 输入是一维数组list,需要转换成numpy支持的二维矩阵进入运算
        list_in = np.array(list_in, ndmin=2)  # ndmin确定最少的维度2，[1, nodes_in]，还需要转置变换
        list_in = list_in.T
        list_out = np.array(list_out, ndmin=2).T  # 同时对list_out做变换，便于比较结果
        hide_in = np.dot(self.wih, list_in)  # 要wih * list_in,那么list_in.shape = [nodes_in, 1]
        hide_out = self.activation(hide_in)
        out_in = np.dot(self.who, hide_out)
        out_out = self.activation(out_in)  # 以上过程与infer过程一致
        # 获取误差
        difference = out_out - list_out  # 计算结果与真值差值
        '''
                grad_who = (ao1-y) * zo1 * ah1
        shape:   [o, h]     [o,1]   [o,1]  [h,1]
                grad_who = (ao1-y) * zo1 * ah1.T
        grad_who.shape=[nodes_out, nodes_hide]
        difference.shape=[nodes_out, 1], out_in.shape=[nodes_out, 1]  shape相同可以用*,对位相乘
        hide_out.shape=[nodes_hide, 1]  此处需要转置方可得到结果矩阵
        '''
        grad_who = np.dot(difference * out_out * (1 - out_out), hide_out.T)
        self.who -= self.lr * grad_who
        '''
                grad_wih = (ao1-y) * zo1 * (1-zo1) * who * zh1 * (1-zh1) * x1
        shape:   [h, i]     [o,1]   [o,1]   [o,1]   [o,h] [h,1]   [h,1]   [i,1]
                grad_wih = who.T DOT {(ao1-y) * zo1 * (1-zo1)} * {zh1 * (1-zh1)} DOT x1.T
        '''
        grad_wih = np.dot(np.dot(self.who.T, difference * out_out * (1 - out_out)) * hide_out * (1 - hide_out), list_in.T)
        self.wih -= self.lr * grad_wih
        pass

    def infer(self, input):
        hide_in = np.dot(self.wih, input)  # 需要接受输入数据[，依次经过各层及激活函数，定义需要加input变量
        hide_out = self.activation(hide_in)  # 输入加权求和后进入激活函数，激活函数不断调用可以补充在初始化中
        out_in = np.dot(self.who, hide_out)
        out_out = self.activation(out_in)
        return out_out
        pass


# 网络实例化，传参。输入节点数根据图片确定，中间层神经元节点数确定的最好办法是实验，不停的选取各种数量，使得网络表现最好。
epochs = 25
nodes_in, nodes_hide, nodes_out, learning_rate = 28*28, 500, 10, 0.05
model = NeuralNetwork(nodes_in, nodes_hide, nodes_out, learning_rate)  # 图片是28*28转一维， 中间层任意，输出层为10个数字
# 导入训练样本
file1 = open('dataset/mnist_train.csv')
data_train = file1.readlines()
file1.close()
for e in range(epochs):
    for list_train in data_train:
        train_features_all =list_train.split(',')
        train_features = np.asfarray(train_features_all[1:], np.float32) / 255.0 * 0.99 + 0.01
        train_labels = np.zeros([nodes_out, ]) + 0.01
        train_labels[int(train_features_all[0])] = 0.99
        print(train_labels)
        model.train(train_features, train_labels)

# 导入测试样本
file2 = open('dataset/mnist_test.csv')
data_test = file2.readlines()
file2.close()
score = []

for list_test in data_test:
    test_features = list_test.split(',')
    true_number = int(test_features[0])  # 读入图片真是结果
    print('真实该数字为：', true_number)
    test_features = (np.asfarray(test_features[1:])) / 255 * 0.99 + 0.01
    test_labels = np.zeros([nodes_out, ]) + 0.01
    test_labels[int(test_features[0])] = 0.99
    pred_labels = model.infer(test_features)
    print(pred_labels)
    pred_number = np.argmax(pred_labels)  # 寻找数组最大值对应的‘编号’，编号即为对应预测的数字
    print('预测该数字为：', pred_number)
    if pred_number == true_number:
        score.append(1)
    else:
        score.append(0)
sum = np.sum(score)
total = np.array(score).shape[0]
accuracy = sum / total
print('准确率为：', accuracy)







