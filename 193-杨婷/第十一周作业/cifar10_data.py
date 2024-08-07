# 此文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf

num_classes = 10
# 设定用于训练和评估的样本总数
train_num = 50000
test_num = 10000

'''
创建空类
这样可以在实例化类后需要定义什么属性就加什么属性，当然也可以提前把所有属性定义到类里再实例化
'''


class CIFAR10Record(object):
    pass


def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1  # cifar10只有10分类，用1位数字就可以表示，如果是Cifar-100数据集，则此处为2
    result.height = 32
    result.weight = 32
    result.depth = 3  # 因为是RGB三通道，所以深度是3

    image_bytes = result.height * result.weight * result.depth  # 图片样本总元素数量
    record_bytes = image_bytes + label_bytes  # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值
    # 使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)  # 使用该类的read()函数从文件队列里面读取文件
    record_bytes = tf.decode_raw(value, tf.uint8)  # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    print(record_bytes)
    print(record_bytes.shape)
    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    # 这里不转换为int32也可以，因为标签为0-9不会超出uint8范围
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    major_depth = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]),
                             [result.depth, result.height, result.weight])
    # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image = tf.transpose(major_depth, [1, 2, 0])

    return result


def inputs(data_dir, batch_size, distorted):
    # 这些文件名是CIFAR-10数据集的五个训练批次文件,它们被假定存储在data_dir指定的目录下。每个文件包含10000个图像及其对应的标签
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]  # 拼接地址,构建跨平台的文件路径
    file_queue = tf.train.string_input_producer(filenames)  # 根据已经有的文件地址创建一个文件队列
    read_input = read_cifar10(file_queue)  # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将格式为uint8的图片数据转换为精度更高的float32
    num_examples_per_epoch = train_num  # train_num事先已定义50000

    if distorted != None:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])  # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        flipped_image = tf.image.random_flip_left_right(cropped_image)  # 将剪切好的图片进行左右翻转
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)  # 将左右翻转好的图片进行随机亮度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)  # 将亮度调整好的图片进行随机对比度调整
        float_image = tf.image.per_image_standardization(adjusted_contrast)  # 进行标准化图片操作，对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24, 24, 3])  # 设置图片数据及标签的形状
        read_input.label.set_shape([1])

        # 决定了在开始训练之前，队列应该至少包含测试集大小的40%的图像数量。
        # 这样做的目的是确保在训练开始之前，队列中有足够的数据可以供模型进行初步的数据处理和计算，从而避免训练过程因等待数据而暂停。
        min_queue_exmaples = int(test_num*0.4)
        # 提醒用户，这个填充过程可能需要几分钟的时间，因为需要加载和处理这些图像数据。
        print('Filling queue with %d CIFAR images before starting to run. This will take a few minutes'
              % min_queue_exmaples)
        '''
        tf.train.shuffle_batch函数将输入的数据（图像和标签）打包成批次，并在每个epoch开始时打乱顺序。
        batch_size参数指定了每个批次的大小。
        num_threads参数指定了用于数据批处理的线程数。
        capacity参数指定了队列的最大容量。
        min_after_dequeue参数指定了在开始训练之前队列中必须保持的最小元素数量，这有助于确保数据的随机性。
        '''
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size, num_threads=16,
                                                            capacity=min_queue_exmaples+3*batch_size,
                                                            min_after_dequeue=min_queue_exmaples)
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        return images_train, tf.reshape(labels_train, [batch_size])
        # 返回值的label的shape为(batchsize,1)，reshape后变成一维(batchsize,)

    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)  # 如果已经大小24*24不会改变，大了会裁剪，小了会填充
        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch*0.4)

        # 这里使用batch()函数代替tf.train.shuffle_batch()函数,无需打乱
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples+3*batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])


















