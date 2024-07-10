import tensorflow as tf
import numpy as np

# 生成模拟数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义占位符
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])


# 定义神经网络结构
def neural_net(x, weights, biases):
    # 隐藏层
    hidden = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden = tf.nn.relu(hidden)
    # 输出层
    out = tf.add(tf.matmul(hidden, weights['h2']), biases['b2'])
    return out


# 初始化权重和偏置
weights = {
    'h1': tf.Variable(tf.random.normal([1, 10])),
    'h2': tf.Variable(tf.random.normal([10, 1]))
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[1, 10])),
    'b2': tf.Variable(tf.constant(0.1, shape=[1, 1]))
}

# 构建神经网络
pred = neural_net(x, weights, biases)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.optimizers.Adam(0.1)

train = optimizer.minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        if step % 50 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}))

    # 作出预测
    print("Prediction:", sess.run(pred, feed_dict={x: [[-1]]}))