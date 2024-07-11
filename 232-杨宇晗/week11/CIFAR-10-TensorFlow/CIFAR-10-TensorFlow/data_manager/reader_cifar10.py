import tensorflow as tf
import cv2

"""通过TensorFlow对TFRecord打包过的数据进行解析"""
#定义要读取的tfrecord路径
filelist = ['data/train.tfrecord']

#定义文件队列
file_queue = tf.train.string_input_producer(filelist,
                                            num_epochs=None,
                                            shuffle=True)
#定义TF文件读取器
reader = tf.TFRecordReader()
#读取文件队列中的数据（序列化之后的数据需解码）
_, ex = reader.read(file_queue)

#定义feature，序列化的格式
feature = {

	#image:以byte型存储，直接解码为string型
    'image':tf.FixedLenFeature([], tf.string),
	
	#label:int型
    'label':tf.FixedLenFeature([], tf.int64)
}

batchsize = 2
"""
通过shuffle_batch定义一个batchsize的数据
	[ex] 读取出来的文件队列中的数据
	batchsize 每个batch中训练样本的数量
	capacity 队列容量
	min_after_dequeue 最小的队列容量

batch 返回一个batchsize的数据
"""
batch  = tf.train.shuffle_batch([ex], batchsize, capacity=batchsize*10,
                       min_after_dequeue=batchsize*5)

"""
对batchsize的数据进行解码
(batchsize的数据,解码时数据的格式)
"""
example = tf.parse_example(batch, features=feature)

#从example中获取 image和label
image = example['image']
label = example['label']

#将image由string转化成byte型即uint8
image = tf.decode_raw(image, tf.uint8)
#对图片数据进行reshape，将它由向量转换成32*32*3的格式
image = tf.reshape(image, [-1, 32, 32, 3])

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
	#通过sess.run读取1个batchsize的数据
    for i in range(1):
        image_bth, _ = sess.run([image,label])
		#利用opencv对图片进行可视化
        cv2.imshow("image", image_bth[0,...])
        cv2.waitKey(0)


