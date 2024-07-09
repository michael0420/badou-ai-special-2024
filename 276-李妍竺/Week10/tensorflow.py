import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version___)

# 1. 使用numpy生成200个随机点
'''  
np.linspace(start= , stop = ,num= )  
np.newaxis: 增加新的维度    ：用于给1维数据转化成矩阵  
    x[:, np.newaxis]: 给列增加维度  x[np.newaixs, :]: 给行增加维度  
'''
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise  # y= x^2+b

# 2. 定义两个placeholder存放输入数据
'''  
tf.placeholder( dtype, shape=None, name=None）  
tf.matmul() 矩阵相乘  
'''
x = tf.placeholder(tf.float32, [None, 1])  # 生成一列预留的位置
y = tf.placeholder(tf.float32, [None, 1])

# 3. 定义神经网络中间层
w1 = tf.Variable(tf.random_normal([1, 10]))  # 权重
bias1 = tf.Variable(tf.zeros([1, 10]))  # 偏差
w1x_b1 = tf.matmul(x, w1) + bias1  # wx+b
L1 = tf.nn.tanh(w1x_b1)  # 激活函数

# 4. 定义神经网络输出层
w2 = tf.Variable(tf.random.normal([10, 1]))  # 中间层的输出是一行，此处设置w2为1列，相乘得到一个数据
bias2 = tf.Variable(tf.zeros([1, 1]))  # 加一个偏差
w2x_b2 = tf.matmul(L1, w2) + bias2  # wx+b
prediction = tf.nn.tanh(w2x_b2)  # 加入激活函数，输出对数据的预测

# 5. 定义损失函数
'''  
处理回归：  
均值平方差MSE：   tf.losses.mean_squared_error(y,pred)                 / tf.reduce_mean(tf.square(y-pred)平均绝对误差MAE:  maes = tf.losses.absolute_difference(y_true, y_pred) ; maes_loss = tf.reduce_sum(maes)  
Huber loss:      hubers = tf.losses.huber_loss(y_true, y_pred)  ; hubers_loss = tf.reduce_sum(hubers)  

处理分类：  
交叉熵：         logs = tf.losses.log_loss(labels=y, logits=y_pred) ; logs_loss = tf.reduce_mean(logs)                /logs = -y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred) ; logs_loss = tf.reduce_mean(logs)'''
loss = tf.reduce_mean(tf.square(y - prediction))

# 6. 定义反向传播算法（梯度下降）
'''  
梯度下降：train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  
momentum: train_step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)  
adam:  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)  
'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 7. 执行操作
'''  
启动图后, 变量必须先经过`初始化` (init) op 初始化,  
首先必须增加一个`初始化` op 到图中.  
init_op = tf.global_variables_initializer()  
'''
sess = tf.Session()
result = sess.run(tf.global_variables_initializer())  # 初始化
for i in range(2000):  # 训练
    sess.run(train_step, feed_dict={x: x_data, y: y_data})

prediction_value = sess.run(prediction, feed_dict={x: x_data})
print('prediction', prediction_value)

print(result)
sess.close()
'''  
with tf.Session() as sess:  
    # 变量初始化  
    sess.run(tf.global_variables_initializer())    # 训练2000次  
    for i in range(2000):        sess.run(train_step, feed_dict={x: x_data, y: y_data})  
    # 获得预测值  
    prediction_value = sess.run(prediction, feed_dict={x: x_data})    print('prediction', prediction_value)'''

# 8. 画图
plt.figure()
plt.scatter(x_data, y_data)  # 散点是真实值
plt.plot(x_data, prediction_value, 'r-', lw=4)  # 曲线是预测值  'r-':红色实线，lw:线宽
plt.show()