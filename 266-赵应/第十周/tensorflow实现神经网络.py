import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""利用简单神经模型实现线性归回"""
if __name__ == '__main__':
    # 生成随机数据样本 维度(200，1)
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.2, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义输层，不带激活函数，并指定输入形状为(None, 1)
    x_input, y_input = tf.placeholder(tf.float32, [None, 1]), tf.placeholder(tf.float32, [None, 1])
    # 定义中间层，初始化权重及偏置
    weight_l1 = tf.Variable(tf.random.normal([1, 10]))
    bias_l1 = tf.Variable(tf.zeros([1, 10]))
    tmp = tf.matmul(x_input, weight_l1) + bias_l1
    activate_l1 = tf.nn.relu(tmp)

    # 定义输出层
    weight_l2 = tf.Variable(tf.random.normal([10, 1]))
    bias_l2 = tf.Variable(tf.zeros([1, 1]))
    tmp2 = tf.matmul(activate_l1, weight_l2) + bias_l2
    predict = tf.nn.tanh(tmp2)

    # 定义损失函数 均方损失
    loss = tf.reduce_mean(tf.square(y_input - predict))

    # 定义反向传播算法 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 开始训练
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # 喂入数据进行训练
        for i in range(2000):
            session.run(train_step, feed_dict={x_input: x_data, y_input: y_data})

        # 获取预测数据
        predict_data = session.run(predict, feed_dict={x_input: x_data})
        # 实际数据与预测数据对比图
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.scatter(x_data, predict_data)
        plt.show()
