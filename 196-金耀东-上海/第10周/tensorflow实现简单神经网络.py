import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

def synthetic_data(num_examples):
    '''生成测试数据'''
    x = np.linspace(start=-3.0, stop=3.0, num=num_examples).reshape([num_examples, -1])
    noise = np.random.normal(loc=0.0, scale=0.5, size=x.shape)
    y = np.square(x) + noise
    return x.astype(np.float64) , y.astype(np.float64)

def next_batch(features, labels, batch_size):
    '''生成小批量数据，用于训练'''
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(x=indices)

    for i in range(0, num_examples, batch_size):
        batch = indices[i:min(num_examples, i+batch_size)]
        yield features[batch], labels[batch]

def generate_params( num_layers, num_input_dim, num_output_dim, num_units ):
    '''生成模型参数'''
    params_list = []
    for i in range(num_layers):
        if i == 0:
            # 输入层输入维度、输出维度
            num_dim_in , num_dim_out = num_input_dim , num_units
        elif i == num_layers-1:
            # 输出层输入维度、输出维度
            num_dim_in , num_dim_out = num_units , num_output_dim
        else:
            # 中间层输入维度、输出维度
            num_dim_in , num_dim_out = num_units , num_units

        w = tf.Variable(tf.random_normal(shape=[num_dim_in, num_dim_out], dtype=tf.float64), trainable=True)
        b = tf.Variable(tf.zeros(shape=[1, num_dim_out], dtype=tf.float64), trainable=True)
        params_list.append([w,b])
    return params_list

def foward(x, params_list):
    '''神经网络正向传播'''
    y_hidde , num_layers = x , len(params_list)
    for layer in range(num_layers-1):
        # 隐藏层：y = sigmoid(x * w + b)
        y_hidde = tf.nn.sigmoid( tf.matmul(y_hidde, params_list[layer][0]) + params_list[layer][1] )
    return tf.matmul(y_hidde, params_list[num_layers-1][0]) + params_list[num_layers-1][1]

if __name__ == "__main__":
    # 测试数据量
    num_exmples = 100

    # 生成测试数据
    x_train, y_train = synthetic_data(num_examples=num_exmples)

    # 定义placehold用于存储输入数据
    x_place = tf.placeholder(dtype=tf.float64,shape=[None, 1])
    y_place = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    # 生成模型参数
    params_list = generate_params(num_layers=2, num_input_dim=1, num_output_dim=1, num_units=16)

    # 正向传播
    y_hat = foward(x_place, params_list)

    # 损失函数
    loss = tf.losses.mean_squared_error(labels=y_train, predictions=y_hat)

    # 优化函数
    optmizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

    with tf.Session() as sess:
        # 变量初始化
        sess.run( tf.global_variables_initializer() )

        # 开始训练
        for epoch in range(100):
            for x_batch, y_batch in next_batch(features=x_train, labels=y_train, batch_size=int(num_exmples/10)):
                sess.run(optmizer, feed_dict={x_place:x_train, y_place:y_train})
            loss_train = sess.run(loss, feed_dict={x_place:x_train, y_place:y_train})
            print(f"epoch {epoch+1} loss:{loss_train}")

        # 获取预测结果
        y_pred = sess.run( y_hat, feed_dict={x_place:x_train} )

    # 画图
    plt.figure()
    plt.scatter(x=x_train, y=y_train) # 散点图为真实数据
    plt.plot(x_train, y_pred, 'r-') # 红色曲线为拟合结果
    plt.show()