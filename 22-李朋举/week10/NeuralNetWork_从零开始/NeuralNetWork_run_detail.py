import numpy
import matplotlib.pyplot as plt

[1]
# open函数里的路径根据数据存储的路径来设定
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

# 把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
# 第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值 （第一列为标签）
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

# 最外层有10个输出节点
'''
最外层有10个输出节点
创建一个全0数组 +0.01 最小值为0.01 , 浮点数更准确(自定义)
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
'''
onodes = 10
targets = numpy.zeros(onodes) + 0.01
'''
all_values[0] 标签列,  将数字变成one-hot
如：   all_values[0] 是标签 7 
   =》 [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01]
'''
targets[int(all_values[0])] = 0.99  # all_values[0] 标签列
print(targets)  # targets第8个元素的值是0.99，这表示图片对应的数字是7(数组是从编号0开始的).

[2]
'''
根据上述做法，我们就能把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
这里需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
'''
# 初始化网络
input_nodes = 784   # 28*28
hidden_nodes = 100  #
output_nodes = 10   #
learning_rate = 0.3
import NeuralNetWork
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()
# 把数据依靠','区分，并分别读入
for record in trainning_data_list:
    all_values = record.split(',')
    """
    (numpy.asfarray(all_values[1:]))  图片的像素值
    """
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系 targets
    '''
    all_values[0] 标签列,  将数字变成one-hot
    如：   all_values[0] 是标签 7 
       =》 [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01]
    '''
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

[3]
'''
最后我们把所有测试图片都输入网络，看看它检测的效果如何
'''
#读入训练数据
#open函数里的路径根据数据存储的路径来设定
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    # 标签 正确答案
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片 图片的像素值
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的 编号
    label = numpy.argmax(outputs)
    # test_label[label]  -> cat
    print("output reslut is : ", label)
    # print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)

[4]
'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
# 加入epocs,设定网络的训练循环次数
epochs = 10

# 每一张图片训练10次
for e in range(epochs):
    # 把数据依靠','区分，并分别读入 每一张图片训练一次
    for record in trainning_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # 训练的过程中更新了权值
        n.train(inputs, targets)
