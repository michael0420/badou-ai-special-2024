'''
该文件主要负责读取cifar-10的数据以及进行数据预处理
'''

import os
import tensorflow as tf
num_classes = 10

# 设定训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class Cifar10Record():
    pass


# 定义一个读取Cifar-10的函数read_cifar10(),这个函数的目标就是为了读取文件里的内容
def read_cifar10(file_queue):
    result = Cifar10Record()

    label_bytes = 1             # 标签值所需要的byte，如果是Cifar_100,则是2
    result.height = 32
    result.width = 32
    result.depth = 3            # 设置样本的维度32*32*3，（h,w,c)
    # 计算每张图片的像素总量，一个像素需要1byte
    img_bytes = result.height * result.width * result.depth
    # 计算样本所需要元素总量，
    record_bytes = label_bytes + img_bytes
    # 使用tf.FixedLengthRecordReader()创建一个文件读取类，
    # 该类的目的就是为了读取文件，record_bytes=参数表示读取的每个记录的字节长度
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 使用该类的read()函数，从文件队列里读取文件
    result.key, value = reader.read(file_queue)
    # 使用tf.decoda_raw()函数，将读取的内容从字符串转为对应的数据类型,图形为uint8
    record_tmp = tf.decode_raw(value, tf.uint8)
    print(record_bytes)
    # 读取到的数组，第一个元素是标签，可以使用strided-slice()函数，将其提取出来，
    # 并使用tf.cast()将其转化成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_tmp, [0], [label_bytes]), tf.int32)
    # 提取图片数据，数据在数据集的储存形式是d*h*w,要将其转化为[d,h,w]
    depth_major = tf.reshape(tf.strided_slice(record_tmp, [label_bytes], [record_bytes]),
                             [result.depth, result.height, result.width])
    # 在pytorch中，数据读入需要以（h,w,c)的形式，tf中不区分，所以在此处将数据转换成（h,w,c)
    result.uint8img = tf.transpose(depth_major, [1, 2, 0])
    return result


# 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir, batch_size, distorted):
    # 拼接地址
    filenames = [os.path.join(data_dir, f'data_batch_{i}.bin')for i in range(1, 6)]
    # 根据已有的文件地址创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 根据队列读取队列中的数据
    read_input = read_cifar10(file_queue)
    # 将数据转化为float32
    reshaped_img = tf.cast(read_input.uint8img, tf.float32)

    num_examples_per_epoch = num_examples_pre_epoch_for_train
    # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
    if distorted:
        # 使用tf.random_crop()对图片随机裁剪出指定的大小
        cropped_img = tf.random_crop(reshaped_img, [24, 24, 3])
        # 对裁剪后的图片进行左右翻转
        flipped_img = tf.image.random_flip_left_right(cropped_img)
        # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        adjusted_brightness = tf.image.random_brightness(flipped_img, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 进行标准化图片操作，tf.image.per_image_standardization()函数
        # 是对每一个像素减去平均值并除以像素方差
        float_img = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图片数据及标签的形状
        float_img.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        #
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train_test.This will take a few minutes."
              % min_queue_examples)

        # 使用tf.train_test.shuffle_batch()函数随机产生一个batch的image和label
        img_train, label_train = tf.train.shuffle_batch(
            [float_img, read_input.label],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            num_threads=16
        )
        print(label_train)
        return img_train, tf.reshape(label_train, [batch_size])
    else:
        # 不对图像进行增强
        # 使用tf.image.resize_image_with_crop_or_pad调整图像的尺寸，
        # 并在尽可能保持原图宽高比的情况下，使其达到一个指定的目标大小
        resized_img = tf.image.resize_image_with_crop_or_pad(reshaped_img, 24, 24)
        # 进行标准化操作
        float_img = tf.image.per_image_standardization(resized_img)

        float_img.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        img_test, label_test = tf.train.batch(
            [float_img, read_input.label],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=16
        )
        return img_test, tf.reshape(label_test, [batch_size])



