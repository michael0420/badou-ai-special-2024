import os
import tensorflow as tf

num_classes = 10


# 设定用于训练和测试的样本量
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


class CIFAR10Record(object):
    pass


# 读取cifar-10的数据
def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1
    result.height=32
    result.width=32
    result.depth=3
    # 样本总数量
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    # 文件读取类，读取文件内容
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 该数组第一个元素是标签，使用strided_slice()提取数据，再使用cast()改变为int32数值类型
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 剩下这些数据在数据集中的存储形式是d * h * w, 需要将其转化为[d,h,w]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    # 将c,h,w 改为h,w,c
    result.uin8image = tf.transpose(depth_major, [1,2,0])

    return result


def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 根据已经有的文件地址创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uin8image, tf.float32)
    num_examples_pre_epoch = num_examples_pre_epoch_for_train
    if distorted is not None:
        # 图片剪切
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 将图片进行左右翻转
        filpped_image = tf.image.random_flip_left_right(cropped_image)
        # 亮度随机调整
        adjusted_brightness = tf.image.random_brightness(filpped_image, max_delta=0.8)
        # 对比度随机调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 标准化操作
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        # 设置图片数据及标签的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        # 打乱顺序，洗牌
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples)
        return images_train, tf.reshape(labels_train, [batch_size])
    else:
        # 图片剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch * 0.4)
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size,
                                                  num_threads=16,   # 启动16个线程
                                                  capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])


