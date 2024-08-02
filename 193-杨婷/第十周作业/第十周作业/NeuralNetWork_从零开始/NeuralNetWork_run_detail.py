import numpy as np
[1]
'''
读取文件、处理数据（归一化、one-hot）
'''
data_file = open("dataset/mnist_train.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

# 把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
# 第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = np.asfarray(all_values[1:]).reshape((28, 28))

# 设置最外层10个节点
onodes = 10
targets = np.zeros(onodes) + 0.01  # 创建一个大小为10的全0数组，最小值是0.01
targets[int(all_values[0])] = 0.99  # all_values[0]是标签，这里实现one-hot
print(targets)  # #targets第6个元素的值是0.99，这表示图片对应的数字是5(数组是从编号0开始的)

[2]
'''
根据上述做法，我们就能把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
这里需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
'''
inputnodes = 784
hiddennodes = 100
outputnodes = 10
learningrate = 0.3

n = NeuralNetWork(inputnodes,hiddennodes,outputnodes,learningrate)
# 读入训练数据
training_data_file = open('dataset/mnist_train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 把数据依靠','区分，并分别读入
for record in training_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01  # numpy.asfarray函数尝试将输入转换为浮点数数组
    targets = np.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

[3]
'''
最后我们把所有测试图片都输入网络，看看它检测的效果如何
'''
# 读入测试数据
test_data_file = open('dataset/mnist_test.csv')
test_data_list = training_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('correct number is:', correct_number)
    # 预处理数字图片
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的 编号
    label = np.argmax(outputs)
    print("output reslut is : ", label)
    '''
    如果是猫狗分类
    test_label[label]  --> cat
    '''
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = np.array(scores)
print('performance:', scores_array.sum()/scores_array.size)

[4]
'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
epochs = 10  # 加入epocs,设定网络的训练循环次数
for i in range(epochs):  # 所有图片训练十次
    for record in test_data_list:  # 每张图片都训练一次
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01  # numpy.asfarray函数尝试将输入转换为浮点数数组
        targets = np.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)





