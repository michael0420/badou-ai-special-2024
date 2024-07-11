import tensorflow as  tf
import numpy as  np
import matplotlib.pyplot as plt

#np.linspace---生成范围内n个随机点
# np.newaxis---转换为矩阵 x[:, np.newaxis] ，放在后面，会给列上增加维度  x[np.newaxis, :] ，放在前面，会给行上增加维度
x_data=np.linspace(-0.5,0.5,200)
x_data=x_data[:,np.newaxis]
print(x_data.shape)
# np.random.normal --生成符合正态分布的随机数
noise=np.random.normal(0,0.03,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个占位符存放输入数据,[None,1]表示N行1列
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义隐藏层  tf.Variable--定义变量
w1=tf.Variable(tf.random_normal([1,10]))
b1=tf.Variable(tf.zeros([1,1]))
#tf.matmul--矩阵乘
res1=tf.matmul(x,w1)+b1
active_res1=tf.nn.tanh(res1)

#定义输出层
w2=tf.Variable(tf.random_normal([10,1]))
b2=tf.Variable(tf.zeros([1,1]))
res2=tf.matmul(active_res1,w2)+b2
active_res2=tf.nn.tanh(res2)

#定义损失函数
error=tf.reduce_mean(tf.square(y-active_res2))
#定义反向传播算法（梯度下降法 ）
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(error)

#变量初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#训练
for i in range(2000):
    sess.run(train_step,feed_dict={x:x_data,y:y_data})

#推理
res3=sess.run(active_res2,feed_dict={x:x_data})

#画图
plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,res3,'r',linewidth='5')
plt.show()


