import tensorflow as tf
import numpy as np

# 定义两个placeholder存放输入数据
inputs = tf.placeholder(tf.float32, [1, 28*28])
targets = tf.placeholder(tf.float32, [1, 10])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_uniform([28*28, 100], minval=-0.5, maxval=0.5))  # 输入28 输出80
biases_L1 = tf.Variable(tf.zeros([1, 100]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(inputs, Weights_L1) + biases_L1
L1 = tf.nn.sigmoid(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_uniform([100, 10], minval=-0.5, maxval=0.5)) # 输出10
biases_L2 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.sigmoid(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(targets - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 读数据
data_file = open('dataset/mnist_train.csv')
data_list = data_file.readlines()
data_file.close()

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    epochs = 7
    for e in range(epochs):
        for index in range(len(data_list)):
            values = data_list[index].split(',')
            #  去掉标签 转成矩阵 归一化
            inputs_data = (np.asfarray(values[1:]))/255.0
            #  转成0.01 和 0.99
            targets_data = np.zeros(10) + 0.01
            targets_data[int(values[0])] = 0.99
            sess.run(train_step, feed_dict={inputs: np.array(inputs_data, ndmin=2), targets: np.array(targets_data, ndmin=2)})

    # 训练完成测试
    data_test_file = open('dataset/mnist_test.csv')
    data_test_list = data_test_file.readlines()
    data_test_file.close()

    success = 0
    for index in range(len(data_test_list)):
        value = data_test_list[index].split(',')
        correct = int(value[0])
        print("正确数字为:", correct)
        inputs_data = (np.asfarray(value[1:])) / 255.0
        # 获得预测值
        prediction_value = sess.run(prediction, feed_dict={inputs: np.array(inputs_data, ndmin=2)})
        number = np.argmax(prediction_value)
        print("推理的数字为:", number)
        if number == correct:
            success += 1
    print("正确率为:", success / 10.0)