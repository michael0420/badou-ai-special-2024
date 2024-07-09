# 该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"


# # 创建一个variable_with_weight_loss()函数，该函数的作用是：
# #   1.使用参数w1控制L2 loss的大小
# #   2.使用函数tf.nn.l2_loss()计算权重L2 loss
# #   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
# #   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
# def variable_with_weight_loss(shape, stddev, w1):
#     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if w1 is not None:
#         weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
#         tf.add_to_collection("losses", weights_loss)
#     return var
#
#
# # 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# # 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
# images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
# images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)
#
# # 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# # 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
# x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
# y_ = tf.placeholder(tf.int32, [batch_size])
#
# # 创建第一个卷积层 shape=(kh,kw,ci,co)
# kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
# bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
#
# # 创建第二个卷积层
# kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
# conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
# bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
# relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
# pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
#
# # 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
# reshape = tf.reshape(pool2, [batch_size, -1])  # 这里面的-1代表将pool2的三维结构拉直为一维结构
# dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后的第二个维度的值
#
# # 建立第一个全连接层
# weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
# fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
#
# # 建立第二个全连接层
# weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
# local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)
#
# # 建立第三个全连接层
# weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
# fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
# result = tf.add(tf.matmul(local4, weight3), fc_bias3)
#
# # 计算损失，包括权重参数的正则化损失和交叉熵损失
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
#
# weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
# loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
#
# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
#
# # 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
# top_k_op = tf.nn.in_top_k(result, y_, 1)
#
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
#     tf.train.start_queue_runners()
#
#     # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
#     for step in range(max_steps):
#         start_time = time.time()
#         image_batch, label_batch = sess.run([images_train, labels_train])
#         _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
#         duration = time.time() - start_time
#
#         if step % 100 == 0:
#             examples_per_sec = batch_size / duration
#             sec_per_batch = float(duration)
#             print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
#                 step, loss_value, examples_per_sec, sec_per_batch))
#
#     # 计算最终的正确率
#     num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
#     true_count = 0
#     total_sample_count = num_batch * batch_size
#
#     # 在一个for循环里面统计所有预测正确的样例个数
#     for j in range(num_batch):
#         image_batch, label_batch = sess.run([images_test, labels_test])
#         predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
#         true_count += np.sum(predictions)
#
#     # 打印正确率信息
#     print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
#
#
#

# # 第一次
# max_steps = 4000
# batch_size = 100
# num_examples_for_eval = 10000
# data_dir = 'Cifar_data/cifar-10-batches-bin'
#
#
# def variable_with_weight_loss(shape, stddev, w1):
#     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if w1 is not None:
#         weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
#         tf.add_to_collection('losses', weights_loss)
#     return var
#     pass
#
#
# # 读取文件
# images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir
#                                                  , batch_size=batch_size, distorted=True)
# images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir
#                                                , batch_size=batch_size
#                                                , distorted=None)
# # 图像数据
# x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
# # 图像标签
# y = tf.placeholder(tf.int32, shape=[batch_size])
#
# # 创建第一个卷积层
# kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# # 这里的x不能直接传image_data
# conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
# bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1), 'relu1')
# pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 创建第二个卷积层
# kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
# conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
# bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
# relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2), name='relu2')
# pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 因为要进行全连接操作，所以这里需要把pool2的输出reshape为一维张量
# reshape = tf.reshape(pool2, shape=[batch_size, -1])
# dim = reshape.get_shape()[1].value
# print(reshape.shape)
#
# # 建立第一个全连接层
# weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
# fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
#
# # 建立第二个全连接层
# weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
# fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)
#
# # 建立第三个全连接层
# weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
# fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
# result = tf.matmul(fc_2, weight3) + fc_bias3
#
# # 计算损失，包括权重参数的正则化损失和交叉熵损失
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))
# weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
# loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
#
# # 误差反向传播更新权值
# train_op = tf.train.AdamOptimizer(1e-3, ).minimize(loss)
#
# top_k_op = tf.nn.in_top_k(result, y, 1)
#
# # 真正地训练
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     tf.train.start_queue_runners()
#     for step in range(max_steps):
#         start_time = time.time()
#         image_batch, label_batch = sess.run([images_train, labels_train])
#         _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch})
#         duration = time.time() - start_time
#
#         if step % 100 == 0:
#             examples_per_sec = batch_size / duration
#             sec_per_batch = float(duration)
#             print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
#                 step, loss_value, examples_per_sec, sec_per_batch))
#
#     # 计算准确度
#     num_batch = int(math.ceil(num_examples_for_eval / batch_size))
#     true_count = 0
#     total_sample_count = num_batch * batch_size
#
#     for j in range(num_batch):
#         image_batch, label_batch = sess.run([images_test, labels_test])
#         predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
#         true_count += np.sum(predictions)
#
#     print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
#


# # 第二次
# def variable_with_weight_loss(shape, stddev, w1):
#     # 生成一个随机的截断weight矩阵
#     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if w1 is not None:
#         # tensorflow1.x中的写法
#         weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
#         # tensorflow2.x中的写法
#         # 这里有问题吗？？？
#         # weights_loss = tf.multiply(tf.reduce_mean(tf.square(var)), w1, name='weights_loss')
#         tf.add_to_collection('losses', weights_loss)
#     return var
#
#
# images_train, labels_train = Cifar10_data.inputs(data_dir, batch_size, distorted=True)
# images_test, labels_test = Cifar10_data.inputs(data_dir, batch_size, distorted=None)
#
# # 这里的数据类型怎么确定？
# x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
# y = tf.placeholder(tf.int32, shape=[batch_size])
#
# # 创建第一个卷积层
# kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# conv1 = tf.nn.conv2d(x, filter=kernel1, strides=[1, 1, 1, 1], padding='SAME')
# bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 创建第二个卷积层
# kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
# conv2 = tf.nn.conv2d(pool1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME')
# bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
# relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
# pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 将得到的多维张量拍扁成一维张量
# reshape = tf.reshape(pool2, shape=[batch_size, -1])
# dim = reshape.get_shape()[1].value
#
# # 创建第一个全连接层
# weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
# fc_1 = tf.matmul(reshape, weight1) + fc_bias1
# fc_1 = tf.nn.relu(fc_1)
#
# # 创建第二个全连接层
# weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
# fc_2 = tf.matmul(fc_1, weight2) + fc_bias2
# fc_2 = tf.nn.relu(fc_2)
#
# # 创建第三个全连接层
# weight3 = variable_with_weight_loss(shape=[192, 10], stddev=0.04, w1=0.004)
# fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
# result = tf.matmul(fc_2, weight3) + fc_bias3
#
# # 计算交叉熵损失
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
#                                                                labels=tf.cast(y, dtype=tf.int64))
# # 权重的L2正则化损失是一种通过限制模型权重大小来防止过拟合的技术，它通过在损失函数中添加一
# # 个与权重平方和成正比的项来实现。
# # 计算权重的正则化损失
# weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
# loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
#
# # 通过反向传播更新权重
# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
#
# # top_k_op = tf.nn.in_top_k(predictions=tf.nn.softmax(result), targets=y, k=1)
# top_k_op = tf.nn.in_top_k(predictions=result, targets=y, k=1)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     tf.train.start_queue_runners()
#     for step in range(max_steps):
#         start_time = time.time()
#         image_batch, label_batch = sess.run(fetches=[images_train, labels_train])
#         _, loss_val = sess.run(fetches=[train_op, loss], feed_dict={x: image_batch, y: label_batch})
#         duration = time.time() - start_time
#         if step % 100 == 0:
#             examples_per_sec = batch_size / duration
#             sec_per_batch = float(duration)
#             print('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)' %
#                   (step, loss_val, examples_per_sec, sec_per_batch))
#     num_batch = int(math.ceil(num_examples_for_eval / batch_size))
#     true_count = 0
#     total_sample_count = num_batch * batch_size
#     for j in range(num_batch):
#         # 这里的label_batch曾经写成了image_label,不要再写错了
#         image_batch, label_batch = sess.run(fetches=[images_test, labels_test])
#         predictions = sess.run(fetches=[top_k_op], feed_dict={x: image_batch, y: label_batch})
#         true_count += np.sum(predictions)
#     print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
#     # 计算最终的正确率
#     num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
#     true_count = 0
#     total_sample_count = num_batch * batch_size


# # 第三次
# def variable_with_weight_loss(shape,stddev,w1):
#     var=tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev))
#     if w1 is not None:
#         weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weights_loss')
#         tf.add_to_collection('losses',weights_loss)
#     return var
#
# # 读取数据
# images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,
#                                               distorted=True)
# images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,
#                                             distorted=True)
# # 定义x和y
# x=tf.placeholder(dtype=tf.float32,shape=[batch_size,24,24,3])
# # tf.int64可以吗？
# y=tf.placeholder(dtype=tf.int64,shape=[batch_size])
#
# # 定义第一个卷积层
# kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
# conv1=tf.nn.conv2d(x,filter=kernel1,strides=[1,1,1,1],padding='SAME')
# bias1=tf.Variable(tf.constant(0.0,shape=[64]))
# relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
# pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
# # 定义第二个卷积层
# kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
# conv2=tf.nn.conv2d(pool1,filter=kernel2,strides=[1,1,1,1],padding='SAME')
# bias2=tf.Variable(tf.constant(0.0,shape=[64]))
# relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
# pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
# # 多维张量变一维
# reshape=tf.reshape(pool2,shape=[batch_size,-1])
# dim=reshape.get_shape()[1].value
# # 定义第一个全连接层
# weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
# fc_bias1=tf.Variable(tf.constant(0.0,shape=[384]))
# fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)
# # 定义第二个全连接层
# weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
# fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
# fc_2=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)
# # 定义第三个全连接层
# weight3=variable_with_weight_loss(shape=[192,10],stddev=0.04,w1=0.004)
# fc_bias3=tf.Variable(tf.constant(0.0,shape=[10]))
# result=tf.matmul(fc_2,weight3)+fc_bias3
#
# # 计算损失（交叉熵损失和权重正则化损失）
# cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
#                                                              labels=tf.cast(y,tf.int64))
# weights_with_l2_loss=tf.add_n(tf.get_collection('losses'))
# loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
#
# # 误差反向传播更新权重
# train_op=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
#
# top_k_op=tf.nn.in_top_k(tf.nn.softmax(result),targets=y,k=1)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     tf.train.start_queue_runners()
#     for step in range(max_steps):
#         image_batch,label_batch=sess.run(fetches=[images_train,labels_train])
#         _,val_loss=sess.run(fetches=[train_op,loss],feed_dict={x:image_batch,y:label_batch})
#         if step%100==0:
#             print(f'step {step},loss={val_loss}')
#     # 检查准确率
#     num_batch=int(num_examples_for_eval/batch_size)
#     total_sample_count=num_batch*batch_size
#     true_count=0
#     for j in range(num_batch):
#         image_batch,label_batch=sess.run(fetches=[images_test,labels_test])
#         top_k=sess.run(fetches=[top_k_op],feed_dict={x:image_batch,y:label_batch})
#         true_count+=np.sum(top_k)
#
#     print(f'accuracy={true_count/total_sample_count*100}%')
#

#
# # 第四次
# # 随机生成权重矩阵
# def variable_with_weight_loss(shape, stddev, w1):
#     var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
#     if w1 is not None:
#         # 使用 tf.nn.l2_loss 函数计算变量 var 的 L2 范数（即所有元素的平方和的平方根，
#         # 但 tf.nn.l2_loss 直接返回平方和，因此实际上计算的是未开方的 L2 范数的平方）。
#         # 将 L2 范数与 w1 相乘，得到权重衰减项 weight_loss。这个权重衰减项表示了变量
#         # var 对整体损失函数的贡献。
#         # 使用 tf.add_to_collection 函数将 weight_loss 添加到名为 'losses' 的集合中。
#         # 在 TensorFlow 中，集合是一种将资源（如变量、张量等）组织在一起的机制，'losses'
#         # 集合通常用于收集模型中的所有损失项，以便在训练时能够方便地汇总它们。
#         weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
#         tf.add_to_collection('losses', weight_loss)
#     return var
#
#
# # 读取数据
# images_train, labels_train = Cifar10_data.inputs(data_dir, batch_size=batch_size,
#                                                  distorted=True)
# images_test, labels_test = Cifar10_data.inputs(data_dir, batch_size=batch_size,
#                                                distorted=None)
#
# # 创建x，y占位符
# x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 24, 24, 3])
# y = tf.placeholder(dtype=tf.int64, shape=[batch_size])
#
# # 构建第一个卷积层，包括卷积，加偏置，relu，池化
# kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
# bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 构建第二个卷积层
# kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
# conv2 = tf.nn.conv2d(pool1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME')
# bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
# relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
# pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # 将多维张量变成一维张量
# reshape = tf.reshape(pool2, [batch_size, -1])
# dim = reshape.get_shape()[1].value
#
# # 构建第一个全连接层
# weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
# fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
#
# # 构建第二个全连接层
# weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
# fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)
#
# # 构建第三个全连接层
# weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
# fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
# result = tf.matmul(fc_2, weight3) + fc_bias3
#
# # 计算损失
# # 这里计算要注意一下，不要搞反了
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))
# weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
# loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
#
# # 通过误差反向传播更新权值
# train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
#
# top_k_op = tf.nn.in_top_k(tf.nn.softmax(result), y, 1)
#
# # 训练
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     tf.train.start_queue_runners()
#     for step in range(max_steps):
#         image_batch, label_batch = sess.run(fetches=[images_train, labels_train])
#         # 这里的val_loss一定不能写成loss，否则会下一次进入循环的时候参数fetches=[train_op,loss]会
#         # 报错，因为loss被改变了
#         _, val_loss = sess.run(fetches=[train_op, loss], feed_dict={x: image_batch, y: label_batch})
#         if step % 100 == 0:
#             print(f'step {step},loss={val_loss}')
#
#     num_batch = int(num_examples_for_eval / batch_size)
#     total_sample_count = num_batch * batch_size
#     true_count = 0
#     for j in range(num_batch):
#         image_batch, label_batch = sess.run(fetches=[images_test, labels_test])
#         top_k = sess.run(fetches=[top_k_op], feed_dict={x: image_batch, y: label_batch})
#         true_count += np.sum(top_k)
#     print(f'accuracy={true_count / total_sample_count * 100}%')


# 第五次
# 生成权重矩阵
def variable_with_weight_loss(shape, stddev, w1):
    # 随机创建一个权重数组
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
    if w1 != None:
        # tf.nn.l2_loss(var)计算的是变量var（通常是一个权重矩阵或向量）的L2范数的平方，
        # 即该变量中所有元素的平方和。这个值作为正则化项，用于惩罚模型中权重的大小，以减少过拟合的风险。
        # 将tf.nn.l2_loss(var)与w1相乘的原因是为了调整正则化项的强度。w1是一个超参数，
        # 它允许我们在训练过程中控制正则化项对总损失函数的贡献程度。具体来说：
        # 如果w1较大，那么正则化项在总损失函数中的权重就会增加，这会导致模型在训练过程中更加倾向于选择较小的权重值，从而可能减少过拟合的风险。
        # 如果w1较小，那么正则化项对总损失函数的影响就会减弱，模型在选择权重时受到的约束就会减少。
        # 如果w1为0，那么正则化项就会被完全忽略，相当于没有应用L2正则化。
        # 因此，通过调整w1的值，我们可以灵活地控制正则化项的强度，以找到最适合我们数据的模型复杂度。
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1)
        tf.add_to_collection('losses', weight_loss)
    return var


# 读取数据
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 定义占位符
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 24, 24, 3])
y = tf.placeholder(dtype=tf.int32, shape=[batch_size])

# 第一个卷积
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(input=x, filters=kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[64]))
relu1 = tf.nn.relu(conv1 + bias1)
pool1 = tf.nn.max_pool2d(input=relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二个卷积
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(input=pool1, filters=kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]))
relu2 = tf.nn.relu(conv2 + bias2)
pool2 = tf.nn.max_pool2d(input=relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# reshape
reshaped = tf.reshape(pool2, shape=[batch_size, -1])
dim = reshaped.shape[1]

# 第一个全连接
weight1 = variable_with_weight_loss(shape=[dim, 512], stddev=0.04, w1=0.004)
bias1 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]))
wx_plus_b_1 = tf.matmul(reshaped, weight1) + bias1
fc_1 = tf.nn.relu(wx_plus_b_1)

# 第二个全连接
weight2 = variable_with_weight_loss(shape=[512, 512], stddev=0.04, w1=0.004)
bias2 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]))
wx_plus_b_2 = tf.matmul(fc_1, weight2) + bias2
fc_2 = tf.nn.relu(wx_plus_b_2)

# 第三个全连接
weight3 = variable_with_weight_loss(shape=[512, 10], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[10]))
wx_plus_b_3 = tf.matmul(fc_2, weight3) + bias3
result = tf.nn.relu(wx_plus_b_3)

# 计算损失
# 包括正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

# 误差反向传播
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# 取出前top_k
top_k_op = tf.nn.in_top_k(tf.nn.softmax(result), y, k=1)

# 训练+预测
# epochs=5
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(epochs):
#         image_batch,label_batch=sess.run(fetches=[images_train,labels_train])
#         sess.run(fetches=[train_op],feed_dict={x:image_batch,y:label_batch})
#
#     image_batch, label_batch = sess.run(fetches=[images_test, labels_test])
#     top_k=sess.run(fetches=top_k_op,feed_dict={x:image_batch,y:label_batch})
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    for step in range(max_steps):
        image_batch, label_batch = sess.run(fetches=[images_train, labels_train])
        # 这里的val_loss一定不能写成loss，否则会下一次进入循环的时候参数fetches=[train_op,loss]会
        # 报错，因为loss被改变了
        _, val_loss = sess.run(fetches=[train_op, loss], feed_dict={x: image_batch, y: label_batch})
        if step % 100 == 0:
            print(f'step {step},loss={val_loss}')

    num_batch = int(num_examples_for_eval / batch_size)
    total_sample_count = num_batch * batch_size
    true_count = 0
    for j in range(num_batch):
        image_batch, label_batch = sess.run(fetches=[images_test, labels_test])
        top_k = sess.run(fetches=[top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(top_k)
    print(f'accuracy={true_count / total_sample_count * 100}%')
