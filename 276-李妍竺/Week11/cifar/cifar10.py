# 神经网络结构构造，训练、评估
import cifar10_data
import tensorflow as tf
import numpy as np
import time
import math

max_steps=4000  #最大次数
batch_size=100
num_examples_for_eval=10000    #评估数据集张数
data_dir="Cifar_data/cifar-10-batches-bin"  #路径

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，

'''
tf.truncated_normal(shape,mean,stddev,dtype,seed,name)
截断方式生成正态分布随机值。截断：随机数值与均值的差不能大于两倍中误差
shape:维度
mean:均值，默认为0
stddev：标准差
'''
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))   #随机出一个shape大小的权重矩阵
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

#读取数据，train:图像增强， test:不进行图像增强
images_train,labels_train=cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True) #需要数据增强
images_test,labels_test=cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x和y_两个placeholder
x=tf.placeholder(tf.float32,[batch_size,24,24,3])  #输入数据
y_=tf.placeholder(tf.int32,[batch_size])       #输出标签

#创建第一个卷积层 shape=(kh,kw,ci,co)
'''
tf.nn.conv2d(input,filters,strides,padding="SAME",data_format=)
input: NHWC: N batch number 或  NCHW
filter：卷积核
strides:步长 随个数字分别表示，NHWC上的步长
padding: same(补成一样大小)/valid

tf.nn.bias_add(value,bias)

tf.nn.max_pool(value,ksize,strides,padding)
ksize:池化窗口的大小
'''
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)   #卷积核大小的权重矩阵，权重值与输入相乘。权重值就是卷积核每一位的值
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")  #步长为2，池化后x尺寸减半。

#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

# tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
'''
a.get_shape()  
a的数据类型只能是tensor，返回元组。
'''
dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))   #384个输出
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)  #matmul: 矩阵乘  multiply:矩阵点乘

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失
'''
tf.nn.sparse_softmax_cross_entropy_with_logits(logits= ,labels=tf.cast(y_,tf.int64)
logits: shape为[batch_size,number class]  type:float32 /float64
labels:[batch_size] type: int32/int64

tf.cast(x,dtpe,name=None)
将x强制转换成dtype
'''
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))

weights_with_l2_loss=tf.add_n(tf.get_collection("losses")) #从集合中取出变量   loss相加
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss   #reduce_mean:所有值的均值

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)  #定义优化器

# tf.nn.in_top_k()：计算输出结果中top k的准确率，函数默认的k值是1，
top_k=tf.nn.in_top_k(result,y_,1)  # 看result与y是否相等

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    '''
    tf.train.coordinator:创建一个线程管理器对象
    tf.train.start_queue_runners:调用
    '''
    tf.train.start_queue_runners()

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time() #当前时间戳
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})  #_,占位符，表示 train result不输出。
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整  向上取整
    true_count=0
    total_sample_count=num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))