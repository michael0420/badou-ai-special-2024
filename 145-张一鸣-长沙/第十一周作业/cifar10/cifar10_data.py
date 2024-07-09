# coding = utf-8

'''
    导入Cifar10数据并进行数据增强等预处理
'''


import os
import tensorflow as tf

category = 10
# 定义训练和测试的样本数量
train_data = 50000
test_data = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class Cifar10Record(object):
    pass


def load_data(file_queue):
    # 用于读取cifar10数据
    result = Cifar10Record()
    label_bytes = 1     # 如果是Cifar-100数据集，则此处为2
    result.height = 32  # 高
    result.width = 32   # 宽
    result.depth = 3    # 通道数为3

    img_bytes = result.height * result.width * result.depth     # 图片样本总元素数量
    record_bytes = label_bytes + img_bytes
    # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    # 使用tf.FixedLengthRecordReader()创建一个文件读取类，该类的目的就是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 使用该类的read()函数从文件队列里面读取文件
    result.key, value = reader.read(file_queue)

    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，
    # 并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + img_bytes]),
                             [result.depth, result.height, result.width])

    # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.unit8image = tf.transpose(depth_major, [1, 2, 0])

    return result
    # 返回读取的信息

def inputs(data_dir, batch_size, distorted):
    # 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    filename = [os.path.join(data_dir, 'data_batch_%d.bin'%i)for i in range(1, 6)]

    # 根据已经有的文件地址创建一个文件队列
    file_queue = tf.train.string_input_producer(filename)

    # 根据已经有的文件队列使用已经定义好的文件读取函数load_data()读取队列中的文件
    read_input = load_data(file_queue)

    # 将已经转换好的图片数据再次转换为float32的形式
    reshaped_img = tf.cast(read_input.unit8image, tf.float32)

    examples = train_data

    if distorted != None:
        # 如果预处理函数中的distorted参数不为空值，就进行图片增强处理
        # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        cropped_img = tf.random_crop(reshaped_img, [24, 24, 3])
        # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
        flipped_img = tf.image.random_flip_left_right(cropped_img)
        # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        bright_img = tf.image.random_brightness(flipped_img, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        adjusted_contrast = tf.image.random_contrast(bright_img, lower=0.2, upper=1.8)
        # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        float_img = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图片数据及标签的形状
        float_img.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(examples * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        images_train, labels_train = tf.train.shuffle_batch([float_img, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,)
        return images_train, tf.reshape(labels_train, [batch_size])

    else:
        # 不对图像数据进行数据增强处理
        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_img = tf.image.resize_image_with_crop_or_pad(reshaped_img, 24, 24)
        # 剪切完成以后，直接进行图片标准化操作
        float_img = tf.image.per_image_standardization(resized_img)
        float_img.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(examples * 0.4)

        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        image_test, labels_test = tf.train.batch([float_img, read_input.label],
                                                 batch_size=batch_size,
                                                 num_threads=16,
                                                 capacity=min_queue_examples + 3 * batch_size)
        return image_test, tf.reshape(labels_test, [batch_size])

