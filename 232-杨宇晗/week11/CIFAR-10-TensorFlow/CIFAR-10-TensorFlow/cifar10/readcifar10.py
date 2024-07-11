import tensorflow as tf

def read(batchsize=64, type=1, no_aug_data=1):
    reader = tf.TFRecordReader()
    if type == 0: #train
        file_list = ["data/train.tfrecord"]
    if type == 1: #test
        file_list = ["data/test.tfrecord"]
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    _, serialized_example = reader.read(filename_queue)

    batch = tf.train.shuffle_batch([serialized_example], batchsize, capacity=batchsize * 10,
                                   min_after_dequeue= batchsize * 5)

    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_example(batch, features = feature)

    images = features["image"]

    img_batch = tf.decode_raw(images, tf.uint8)
    img_batch = tf.cast(img_batch, tf.float32)
    img_batch = tf.reshape(img_batch, [batchsize, 32, 32, 3])
	
	"""如果从训练集中读取，则添加数据增强"""
    if type == 0 and no_aug_data == 1:
		#随机裁剪
        distorted_image = tf.random_crop(img_batch,						
                                         [batchsize, 28, 28, 3])	
        #随机对比度
		distorted_image = tf.image.random_contrast(distorted_image,	
                                                   lower=0.8,
												   upper=1.2)
        #随机色调
		distorted_image = tf.image.random_hue(distorted_image,
                                      max_delta=0.2)
        #随机饱和度
		distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.8,
                                                     upper=1.2)
        #对处理过的图像约束取值范围
		img_batch = tf.clip_by_value(distorted_image, 0, 255)
	
    img_batch = tf.image.resize_images(img_batch, [32, 32])
    label_batch = tf.cast(features['label'], tf.int64)

    #将图片的数据范围处理到-1——1之间
    img_batch = tf.cast(img_batch, tf.float32) / 128.0 - 1.0
    #
    return img_batch, label_batch


