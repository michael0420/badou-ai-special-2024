import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os



num_classes = 10

# 设定用于训练和评估样本总数
num_train_examples_for_one_epoch = 50000
num_test_examples_for_one_epoch = 10000


class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result = CIFAR10Record()

    # 由于只要10累 ，所以用1位数
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    images_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes + images_bytes #标签位数加上图片位数

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 利用该类的read（）函数从文件队列中读取文件
    result.key,value = reader.read(file_queue)

    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应像素数组
    record_bytes = tf.decode_raw(value,tf.uint8)

    # 该数组第一个是标签， 用strided函数提取，再用cast把标签转化
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+images_bytes]),[result.depth,result.height,result.width])

    # 我们要将之前分割好的图像数据使用tf.transpose（）函数转化为c h w 顺序后
    # 这一步是转化为h w c
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result




def inputs(data_dir,batch_size,distorted):
    file_name = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6) ]

    # 根据已有的文件创建文件队列
    file_quen = tf.train.string_input_producer(file_name)
    read_input = read_cifar10(file_quen)

    # 将已经转化好的数据再转化格式
    reshape_img = tf.cast(read_input.uint8image,tf.float32)

    num_examples_for_one_epoch = num_test_examples_for_one_epoch + num_train_examples_for_one_epoch

    if distorted !=None:
        #剪贴图片
        cropped_image = tf.random_crop(reshape_img,[24,24,3])
        flipped_image = tf.image.random_flip_left_right(cropped_image) # 将剪贴好的图片左右旋转

        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)

        #对比度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)

        # 标准化操作，对每一像素减去平均值并除以像素方差

        float_image = tf.image.per_image_standardization(adjusted_contrast)

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])


        min_queue_examples = int(num_examples_for_one_epoch * 0.4)

        print('filling queue with %d CIFAR images before starting to train it will take few minutes'%min_queue_examples)

        images_train,label_train = tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                          num_threads = 16,
                                                          capacity = min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue = min_queue_examples
                                                          )


        return  images_train,tf.reshape(label_train,[batch_size])
    else:
        resized_img = tf.image.resize_with_crop_or_pad(reshape_img,24,24)


        float_image = tf.image.per_image_standardization(resized_img)

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_for_one_epoch * 0.4)

        images_test , labels_test = tf.train.batch(
            [float_image,read_input.label],
            batch_size = batch_size,
            num_threads = 16,
            capacity = min_queue_examples + 3* batch_size

        )
        return images_test,tf.reshape(labels_test,[batch_size])
















