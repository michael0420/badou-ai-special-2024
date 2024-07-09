import tensorflow as tf
import cv2
import numpy as np
import glob
classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']
#定义变量，当前遍历到的类别
idx = 0
#定义图片的数据
im_data = []
#定义图片的标签
im_labels = []
#获取classification各个类别下所有的图片
for path in classification:
    path = "data/image/train/" + path
    im_list = glob.glob(path + "/*")
    im_label = [idx for i in  range(im_list.__len__())]
    idx += 1
    im_data += im_list
    im_labels += im_label

print(im_labels)
print(im_data)

#定义文件存放路径
tfrecord_file = "data/train.tfrecord"
#定义tfrecord写入的实例
writer = tf.python_io.TFRecordWriter(tfrecord_file)   

#将list中的索引值通过shuffle打乱
index = [i for i in range(im_data.__len__())]
np.random.shuffle(index)


#遍历当前的数据列表
for i in range(im_data.__len__()):
    im_d = im_data[index[i]]
    im_l = im_labels[index[i]]
	#通过opencv对图片数据进行读取
    data = cv2.imread(im_d)
    
	#通过gfile对图片数据进行读取（数据本身为byte型，文件大小会更小，会对数据进行编码，需要解码）
	#data = tf.gfile.FastGFile(im_d, "rb").read()
    
	ex = tf.train.Example(
        features = tf.train.Features(
            #通过字典的形式存放具体数据
			feature = {
                "image":tf.train.Feature(
					#通过byte型进行存储
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(
                    #通过int型进行存储
					int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )
	#对当前数据进行序列化
    writer.write(ex.SerializeToString())
#关闭weiter
writer.close()
