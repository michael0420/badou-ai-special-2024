import tensorflow as tf

"""通过string_input_producer从文件中读取数据"""

#文件路径
filename = ['data/A.csv', 'data/B.csv', 'data/C.csv']
#通过string_input_producer产生一个文件队列，输出的为文件队列
file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)
"""从文件队列中读取数据"""
#定义一个文件读取器
reader = tf.WholeFileReader()
#对文件进行读取 key:文件名,value:文件内容
key, value = reader.read(file_queue)

with tf.Session() as sess:
	#对计算图中的局部变量初始化
    sess.run(tf.local_variables_initializer())
	
	#构造文件队列填充线程
    tf.train.start_queue_runners(sess=sess)
    
	#从文件队列获取数据
	for i in range(6):
        print(sess.run([key, value]))