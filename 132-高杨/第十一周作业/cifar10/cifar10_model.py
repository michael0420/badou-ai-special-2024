import tensorflow.compat.v1 as tf
import  numpy as np
import time
import math
import cifar10_enhance
tf.disable_v2_behavior()



max_steps = 4000
batch_sizes = 100
num_examples_for_eval = 10000
data_dir = 'Cifar_data/cifar-10-batches-bin'

'''
使用w1控制 L2 LOSS 大小
tf.nn.l2_loss() 计算权重L2 LOSS
tf.multiply()  计算权重L2 LOSS 与w1的乘积 赋值给weight_loss
tf.add_to_collection  将最终结果放在名为losses的集合里方便后续计算
'''

def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not  None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weights_loss')
        tf.add_to_collection('losses',weights_loss)


images_train, labels_train = cifar10_enhance.inputs(data_dir=data_dir,batch_size=batch_sizes,distorted=True)
images_test , labels_test = cifar10_enhance.inputs(data_dir=data_dir,batch_size=batch_sizes,distorted=False)

x = tf.placeholder(tf.float32,[batch_sizes,24,24,3])
y = tf.placeholder(tf.int32,[batch_sizes])

# 创建第一个卷积层 shape = h , k c ,o
kernel1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
baise1 = tf.Variable(tf.constant(0.0,shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,baise1))
pool1 = tf.nn.max_pool(relu1,[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 创建第二个卷积层

kernel2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
baise2 = tf.Variable(tf.constant(0.1,shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,baise2))
pool2 = tf.nn.max_pool(relu2,[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 进行全连接层的操作，所以用tf.reshape() 把pool2 输出变成一纬向量
reshape = tf.reshape(pool2,[batch_sizes,-1])
dim = reshape.get_shape()[1].value

# 第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1,shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape,weight1) + fc_bias1)

# 第二个全连接层
weight2 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1,shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1,weight2) + fc_bias2)

# 第三个全连接层
weight3 = variable_with_weight_loss(shape=[192,10],stddev= 1 / 192.,w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1,shape=[10]))
res = tf.nn.relu(tf.matmul(fc_2,weight3) + fc_bias3)

# 计算损失， 权重参数正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=res,labels=tf.cast(y,tf.int64))

weight_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(res,y,1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启用线程操作， 之前数据增强中我们使用了 多线程
    tf.train.start_queue_runners()

    for setp in range(max_steps):
        start_time = time.time()
        image_batch , label_batch = sess.run([images_train,labels_train])
        _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration = time.time() - start_time

        if setp%100==0:
            examples_per_sec = batch_sizes / duration
            sec_per_batch = float(duration)
            print('step %d , loss=%.2f(%.1f examples / sec; %.3f sec/batch)'%(setp,loss_value,examples_per_sec,sec_per_batch))

# 计算最终正确率

    num_batch = int(math.ceil(num_examples_for_eval / batch_sizes))

    true_count = 0
    total_sample_count = num_batch * batch_sizes

    for j in range(num_batch):
        image_batch,label_batch = sess.run([images_test,labels_test])
        predictions = sess.run([top_k_op],feed_dict={x:image_batch,y:label_batch})

        true_count += np.sum(predictions)

    print('acc:  %.3f'%((true_count / total_sample_count)*100))



