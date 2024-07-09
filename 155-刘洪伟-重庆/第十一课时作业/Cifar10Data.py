# _*_ coding: UTF-8 _*_
# @Time: 2024/7/1 11:40
# @Author: iris
# @Email: liuhw0225@126.com
# 该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf

# 分类总量
num_classes = 10

# 训练样本总数
epoch_for_train = 50000
# 测试样本总数
epoch_for_eval = 10000


class CIFAR10Record(object):
    """
    定义空的类，返回读取的cifar-10数据
    """
    pass


def data_treating(file_queue):
    """
    读取目标文件里面的内容
    :param file_queue: 文件地址
    :return:
    """
    result = CIFAR10Record()
    # 用于处理cifar10/cifar100(label的占位)
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.channel = 3
    # 计算样本总数量
    image_bytes = result.height * result.width * result.channel
    # 样本总数+标签数
    record_bytes = label_bytes + image_bytes
    # 读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 从文件队列中读取文件
    result.key, value = reader.read(file_queue)
    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.channel, result.height, result.width])
    # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs(data_dir, batch_size, distorted):
    """
    读取目标文件的数据，并进行数据增强处理
    :param data_dir:
    :param batch_size:
    :param distorted:
    :return:
    """
    # 拼接地址
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    # 创建文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 读取文件
    read_input = data_treating(file_queue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    num_examples_per_epoch = epoch_for_train

    if distorted is not None:
        # 随机裁剪图像
        cropped_iamge = tf.random_crop(reshaped_image, [24, 24, 3])
        # 随机旋转图像
        flipped_image = tf.image.random_flip_left_right(cropped_iamge)
        # 随机亮度调整（随机缩放图像）
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 随机对比度调整（随机对比度）
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 随机色彩平衡
        random_hue = tf.image.random_hue(adjusted_contrast, max_delta=0.2)
        # 随机裁剪和缩放
        random_crop_and_scale = tf.image.random_saturation(random_hue, lower=0.2, upper=1.8)
        # 随机旋转和缩放
        random_rotation_and_scale = tf.image.rot90(random_crop_and_scale,
                                                   k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # 进行标准化图片操作
        float_image = tf.image.per_image_standardization(random_rotation_and_scale)
        # 设置图片数据及标签的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        return images_train, tf.reshape(labels_train, [batch_size])
    else:
        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)

        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])
