import tensorflow as tf


"""通过slice_input_producer读取文件列表中的样本"""

#定义图片路径list
images = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
#定义label路径
labels = [1, 2, 3, 4]

"""
TensorFlow对数据进行读取时首先创建一个文件队列。
通过slice_input_producer( , , ,)实现，输出的为Tensor

num_epochs =
	1 对所有样本进行1遍循环
	2 对所有样本进行2遍循环
	None 不规定循环次数，只要想读取就有数据

shuffle = 
	True  对当前文件的数据进行打乱
	False 不打乱
"""
[images, labels] = tf.train.slice_input_producer([images, labels],
                              num_epochs=None,
                              shuffle=True)

with tf.Session() as sess:
	#对计算图中的局部变量初始化
    sess.run(tf.local_variables_initializer())
	
	#构造文件队列填充线程
    tf.train.start_queue_runners(sess=sess)
	
	#从文件队列获取数据
    for i in range(10):
        print(sess.run([images, labels]))
