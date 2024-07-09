import os
import tensorflow as tf

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class CIFAR10Record(object):
    pass



def read_cifar10_data(file_queue):
    result = CIFAR10Record()

    # 1为10分类 2为100分类
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.channels = 3

    #图片长度
    image_bytes = result.height * result.width * result.channels

    #图片和标签长度
    record_bytes = label_bytes + image_bytes

    #文件读取器
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)

    result.key,value=reader.read(file_queue)

    #读取的文件是字符串格式，需要转换像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.lable = tf.cast(tf.strided_slice(record_bytes, [0],[label_bytes]), dtype=tf.int32)

    # depth * height * width
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],[record_bytes]),[result.channels, result.height ,result.width])

    #维度转换depth * height * width ->  height * width * depth
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result


def inputs(data_dir,batch_size,distorted):
    fileNames = [os.path.join(data_dir, 'data_batch_%d.bin'%i) for i in range(1, 6)]
    #文件下载队列
    fileQueue = tf.train.string_input_producer(fileNames)
    #读取文件
    read_input = read_cifar10_data(fileQueue)
    #转换图片格式
    reshaped_image = tf.cast(read_input.uint8image,dtype = tf.float32)

    num_examples_per_epoch = num_examples_pre_epoch_for_train

    #图片增强
    if distorted is not None:
        #切割
        cropped_image = tf.random_crop(reshaped_image, [24,24,3])
        #翻转
        left_right_image = tf.image.random_flip_left_right(cropped_image)
        #亮度
        brightness_image = tf.image.random_brightness(left_right_image,max_delta = 0.8)
        #对比度
        contrast_image = tf.image.random_contrast(brightness_image, lower=0.5, upper=1.5)

        #图片标准化
        standard_image = tf.image.per_image_standardization(contrast_image)
        #重新设置图片和标签维度
        standard_image.set_shape([24,24,3])
        read_input.lable.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)
        #图片洗牌
        images_train, labels_train = tf.train.shuffle_batch([standard_image, read_input.lable], batch_size = batch_size,
                               capacity = min_queue_examples + 3 *batch_size,
                               min_after_dequeue = min_queue_examples)

        return standard_image, tf.reshape(labels_train, [batch_size])
    else :
        #图片剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 图片标准化
        standard_image = tf.image.per_image_standardization(resized_image)
        # 重新设置图片和标签维度
        standard_image.set_shape([24, 24, 3])
        read_input.lable.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)

        images_test, labels_test = tf.train.batch([standard_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])