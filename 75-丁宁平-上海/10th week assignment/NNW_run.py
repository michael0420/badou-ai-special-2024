import numpy as np
import matplotlib.pyplot as plt

[1]
with open('dataset/mnist_test.csv') as data_file:
    data_list = data_file.readlines()
    print(len(data_list))
    print(data_list[0])

# 把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
# 第一个值是图片表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = np.asarray(all_values[1:]).reshape((28,28))

# 最外层有10个输出节点
onodes = 10
targets = np.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)

[2]
'''
根据上述做法，我们就能把输入的图片和它对应的数字建立联系，这种联系就可以用于输入到网络中，进行训练
图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个节点。
需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题变化。
确定中间层神经元节点数最好的办法是实验，不停地选取各种数量，看看哪种数量能使得网络表现得最好。
'''
# 初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
with open('dataset/mnist_train.csv') as training_data_file:
    training_data_list = training_data_file.readlines()
# 数据依靠','区分，并分别读入
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]))/ 225.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(input_nodes, targets)

[3]
'''
最后我们把所有测试图片都输入网络，看看它检测的效果如和
'''
scores= []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('该图片对应的数字为：', correct_number)
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:]))/ 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字，推理
    outputs = n.query(inputs)
    找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print('output result is :', label)
    # print（“网络认为图片的数字是：”，label）
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(score)

# 计算图片判断的成功率
score_array = np.asfarray(scores)
print('perfermance = ', score_array.sum()/score_array.size)

[4]
'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题。
'''
# 加入epochs ，设定网络的训练循环次数
epochs = 10

for e in range(epochs)：
    for record in trainning_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]))/ 255.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)