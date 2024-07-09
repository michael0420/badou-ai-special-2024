import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# 读入训练数据
with open('dataset/mnist_train.csv', 'r') as trainfile:
    train_data_list = trainfile.readlines()
trainImages = []
trainLables = []
for record in train_data_list:
    all_values = record.split(',')
    trainImages.append(all_values[1:])
    trainLables.append(all_values[0])
# 数据归一化
trainImages = np.asfarray(trainImages) / 255.0 * 0.999 + 0.001
# 标记进行onehot编码
trainLength = len(trainLables)
targets = np.zeros((trainLength, output_nodes)) + 0.001
for i in range(len(trainLables)):
    value = int(trainLables[i])
    targets[i][value] = 0.999
trainLables = targets
# ======================
# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
x_data = trainImages
y_data = trainLables
# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([784, 5]))
biases_L1 = tf.Variable(tf.zeros([1, 5]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.sigmoid(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([5, 10]))
biases_L2 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.sigmoid(Wx_plus_b_L2)  # 加入激活函数
# prediction = tf.nn.softmax(prediction)  # 加入激活函数

# 定义损失函数（均方差函数）
# loss = tf.reduce_mean(tf.square(y - prediction))
# loss = tf.nn.softmax_cross_entropy_with_logits(y, prediction)
loss = -sum([y[:, i] * tf.log(prediction[:, i]) for i in range(output_nodes)])
# loss = tf.reduce_mean(np.square(y - prediction))#NotImplementedError
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练5次
    for j in range(5):
        for i in range(trainLength):
            sess.run(train_step, feed_dict={x: x_data[i].reshape(1, 784), y: y_data[i].reshape(1, 10)})
    # # 读取数据
    # with open('dataset/mnist_test.csv', 'r') as test_data_file:
    #     test_data_list = test_data_file.readlines()
    # testImages = []
    # testLables = []
    # for record in test_data_list:
    #     all_values = record.split(',')
    #     testImages.append(all_values[1:])
    #     testLables.append(all_values[0])
    # # 数据归一化
    # testImages = np.asfarray(testImages) / 255.0 * 0.999 + 0.001
    # # 标记进行onehot编码
    # targets = np.zeros((len(testLables), output_nodes)) + 0.001
    # for i in range(len(testLables)):
    #     targets[i, int(testLables[i])] = 0.999
    # testLables = targets
    # testLength = len(testLables)
    # x_data = testImages
    # y_data = testLables
    # 预测
    for i in range(5):
        # 获得预测值
        prediction_value = sess.run(prediction, feed_dict={x: x_data[i].reshape(1, 784)})
        # print(prediction_value[:, 1])
        print('预测值为：{}，真实值为{}'.format(np.argmax(prediction_value), np.argmax(y_data[i])))
