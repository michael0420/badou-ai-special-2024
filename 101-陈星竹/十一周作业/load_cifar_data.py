# 2.tf实现cifar10数据集分类
import os
import tensorflow as tf
import numpy as np
import time
import math

num_classes = 10
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000
max_seps = 4000
batch_size = 100
data_dir = 'cifar_data/cifar-10-batches-bin'

#空类 用于储存从文件中读取的数据，包括图像和标签
class CIFAR10Record(object):
    pass

#从指定路径下读取数据集
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1 #表示数据集类别，如果是cifar100 此处为2
    #定义RGB三通道以及长宽
    result.height = 32
    result.width = 32
    result.depth = 3

    img_bytes = result.height*result.width*result.depth #图像的字节数
    record_bytes = label_bytes+img_bytes #每条记录的总字节数

    #tf.FixedLengthRecordReader创建一个文件读取类
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value = reader.read(file_queue) #读取文件

    #读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes = tf.decode_raw(value,tf.uint8)

    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes],
                                            [1]),tf.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    # reshape：改变数据的维度
    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],
                                              [label_bytes+img_bytes],[1]),
                                              [result.depth,result.height,result.width])

    # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c),原来是(c,h,w)对应索引改变位置，所以是[1,2,0]
    # transpose:改变维度的顺序
    result.uint8img = tf.transpose(depth_major,[1,2,0])

    #返回值包含目标文件的所有信息，并且整理好了
    return result

 #这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def input(data_dir,batch_size,distorted):
    #创建包含所有数据批次文件路径的列表
    filenames = [os.path.join(data_dir,"data_batch_%d.bin"%i) for i in range(1,6)]
    #创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件
    read_input = read_cifar10(file_queue)
    # 将已经转换好的图片数据再次转换为float32的形式
    reshaped_img = tf.cast(read_input.uint8img,tf.float32)

    num_examplex_pre_epoch = num_examples_pre_epoch_for_train

    if distorted != None:
        cropped_image = tf.random_crop(reshaped_img,[24,24,3]) #随机裁剪
        flipped_image = tf.image.random_flip_left_right(cropped_image) #随机左右翻转
        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)#调整图片亮度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.5)#随机调整图片对比度
        #tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        float_img = tf.image.per_image_standardization(adjusted_contrast)#对图像进行标准化

        # 设置图片的形状和标签的形状
        float_img.set_shape([24,24,3])
        read_input.label.set_shape([1])

        #最小样本数
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train, labels_train = tf.train.shuffle_batch([float_img, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size, #队列的容量
                                                            min_after_dequeue=min_queue_examples) #出队后队列中至少保留的样本数，
                                                                                                    # 确保数据随机性和队列稳定性

        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train, tf.reshape(labels_train, [batch_size]) #调整标签形状
    else:
        resized_img = tf.image.resize_image_with_crop_or_pad(reshaped_img, 24, 24)
        float_img = tf.image.per_image_standardization(resized_img) # 标准化

        float_img.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        image_test,label_test = tf.train.batch([float_img,read_input.label],batch_size=batch_size,
                                                 capacity=min_queue_examples+batch_size*3,
                                                 num_threads=16)

        return image_test,tf.reshape(label_test,[batch_size])


#定义变量函数
#用于创建变量，并添加 L2 正则化损失
def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weights_loss')
        tf.add_to_collection('losses',weights_loss)
    return var

#读取训练数据和验证数据
images_train,labels_train = input(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test = input(data_dir=data_dir,batch_size=batch_size,distorted=None)

x = tf.placeholder(tf.float32,[batch_size,24,24,3])#图像占位符
y = tf.placeholder(tf.int32,[batch_size])#标签占位符

#第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0) # 第一个卷积核
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64])) #偏置
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.5,shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2 = tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#进入全连接层
# 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，
# 并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2,[batch_size,-1])#压缩成一维数据
dim = reshape.get_shape()[1].value

#第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim,512],stddev=0.04,w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1,shape=[512]))
fc_1 = tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#第二个全连接层
weight2 = variable_with_weight_loss(shape=[512,256],stddev=0.04,w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1,shape=[256]))
fc_2 = tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#第三个全连接层
weight3 = variable_with_weight_loss(shape=[256,10],stddev=0.04,w1=0.004)
fc_bias3 = tf.Variable(tf.constant(0.1,shape=[10]))
result = tf.nn.softmax(tf.matmul(fc_2,weight3)+fc_bias3)

#计算损失
#交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))
#权重损失
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
#优化器选择Adam,包含了反向传播的过程
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#输出分类准确率最高的数值
top_k_op = tf.nn.in_top_k(result,y,1)

#TensorFlow开始前要初始化所有变量
inti_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti_op)
    # 启动数据队列线程，因为再数据预处理中使用了队列
    tf.train.start_queue_runners(sess=sess)

    for step in range(max_seps):
        start_time = time.time()
        image_batch ,label_batch = sess.run([images_train,labels_train])
        _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration = time.time() - start_time

        if step %100 ==0:
            examples_per_sec = batch_size / duration #每秒钟能训练的样本数量
            sec_per_batch = float(duration) #训练一个batch花费的时间
            print("step %d,loss=%.2f, (%.1f examples/sec; %.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

    #正确率
    num_batch = int(math.ceil(num_examples_pre_epoch_for_eval/batch_size))
    true_count = 0
    total_count = num_batch * batch_size

    for j in range(num_batch):
        image_batch,label_batch = sess.run([images_test,labels_test])
        predicted = sess.run([top_k_op],feed_dict={x:image_batch,y:label_batch})
        true_count += np.sum(predicted)

    print("正确率：%.3f%%"%(true_count/total_count)*100)