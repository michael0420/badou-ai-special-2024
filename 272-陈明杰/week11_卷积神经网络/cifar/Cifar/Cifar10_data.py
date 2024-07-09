# 该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf

# num_classes = 10
#
# # 设定用于训练和评估的样本总数
# num_examples_pre_epoch_for_train = 50000
# num_examples_pre_epoch_for_eval = 10000
#
#
# # 定义一个空类，用于返回读取的Cifar-10的数据
# class CIFAR10Record(object):
#     pass
#
#
# # 定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
# def read_cifar10(file_queue):
#     result = CIFAR10Record()
#
#     label_bytes = 1  # 如果是Cifar-100数据集，则此处为2
#     result.height = 32
#     result.width = 32
#     result.depth = 3  # 因为是RGB三通道，所以深度是3
#
#     image_bytes = result.height * result.width * result.depth  # 图片样本总元素数量
#     record_bytes = label_bytes + image_bytes  # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值
#
#     reader = tf.FixedLengthRecordReader(
#         record_bytes=record_bytes)  # 使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
#     result.key, value = reader.read(file_queue)  # 使用该类的read()函数从文件队列里面读取文件
#
#     record_bytes = tf.decode_raw(value, tf.uint8)  # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
#
#     # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
#     result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
#
#     # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
#     # 这一步是将一维数据转换成3维数据
#     depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
#                              [result.depth, result.height, result.width])
#
#     # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
#     # 这一步是转换数据排布方式，变为(h,w,c)
#     result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#
#     return result  # 返回值是已经把目标文件里面的信息都读取出来
#
#
# def inputs(data_dir, batch_size, distorted):  # 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
#     filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]  # 拼接地址
#
#     file_queue = tf.train.string_input_producer(filenames)  # 根据已经有的文件地址创建一个文件队列
#     read_input = read_cifar10(file_queue)  # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件
#
#     reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将已经转换好的图片数据再次转换为float32的形式
#
#     num_examples_per_epoch = num_examples_pre_epoch_for_train
#
#     if distorted != None:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
#         cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])  # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
#
#         flipped_image = tf.image.random_flip_left_right(
#             cropped_image)  # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
#
#         adjusted_brightness = tf.image.random_brightness(flipped_image,
#                                                          max_delta=0.8)  # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
#
#         adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2,
#                                                      upper=1.8)  # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
#
#         float_image = tf.image.per_image_standardization(
#             adjusted_contrast)  # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
#
#         float_image.set_shape([24, 24, 3])  # 设置图片数据及标签的形状
#         read_input.label.set_shape([1])
#
#         min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
#         print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
#               % min_queue_examples)
#
#         images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
#                                                             num_threads=16,
#                                                             capacity=min_queue_examples + 3 * batch_size,
#                                                             min_after_dequeue=min_queue_examples,
#                                                             )
#         # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
#
#         return images_train, tf.reshape(labels_train, [batch_size])
#
#     else:  # 不对图像数据进行数据增强处理
#         resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24,
#                                                                24)  # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
#
#         float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作
#
#         float_image.set_shape([24, 24, 3])
#         read_input.label.set_shape([1])
#
#         min_queue_examples = int(num_examples_per_epoch * 0.4)
#
#         images_test, labels_test = tf.train.batch([float_image, read_input.label],
#                                                   batch_size=batch_size, num_threads=16,
#                                                   capacity=min_queue_examples + 3 * batch_size)
#         # 这里使用batch()函数代替tf.train.shuffle_batch()函数
#         return images_test, tf.reshape(labels_test, [batch_size])


#
# # /////////////////////////////////////////////////////////////////////////////////
# num_classes = 10
# num_examples_pre_epoch_for_train = 50000
# num_examples_pre_epoch_for_eval = 10000
#
#
# class CIFAR10Record(object):
#     pass
#
#
# def read_cifar10(file_queue):
#     read_result = CIFAR10Record()
#     label_bytes = 1  # 如果是cifar-100数据集，则这里是2，因为[0,10)有9个元素，一位就能表示，[0,100)需要两位
#     read_result.high = 32
#     read_result.width = 32
#     read_result.depth = 3
#     image_bytes = read_result.high * read_result.width * read_result.depth
#     record_bytes = image_bytes + label_bytes
#     reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#     read_result.key, value = reader.read(file_queue)
#     record_bytes = tf.decode_raw(value, tf.uint8)
#     read_result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
#     depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
#                              [read_result.depth, read_result.high, read_result.width])
#     read_result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#     return read_result
#
#     pass
#
#
# # def read_cifar10(file_queue):
# #     # 定义一个名为read_cifar10的函数，它接受一个参数file_queue，这通常是一个用于读取TFRecords文件的队列。
# #
# #     read_result = CIFAR10Record()
# #     # 创建一个CIFAR10Record对象，并赋值给read_result变量。CIFAR10Record是一个自定义类，用于存储读取的CIFAR-10数据。
# #
# #     label_bytes = 1  # 如果是cifar-100数据集，则这里是2
# #     # 设置label_bytes为1，表示CIFAR-10的标签使用一个字节存储。对于CIFAR-100，则应为2个字节。
# #
# #     read_result.high = 32
# #     # 将read_result对象的high属性设置为32，但这里通常应该命名为height，表示图像的高度。
# #
# #     read_result.weight = 32
# #     # 将read_result对象的weight属性设置为32，但这里通常应该命名为width，表示图像的宽度。
# #
# #     read_result.depth = 3
# #     # 设置read_result对象的depth属性为3，表示图像的颜色通道数（对于RGB图像）。
# #
# #     image_bytes = read_result.high * read_result.weight * read_result.depth
# #     # 计算图像数据所需的字节数。这里实际上是计算图像的总大小（以字节为单位）。
# #
# #     record_bytes = image_bytes + label_bytes
# #     # 计算每条记录的总字节数，即图像字节数加上标签字节数。
# #
# #     reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
# #     # 创建一个FixedLengthRecordReader对象，用于从file_queue中读取固定长度的记录。
# #
# #     read_result.key, value = reader.read(file_queue)
# #     # 调用reader的read方法从file_queue中读取一条记录，并将记录的键和值分别赋值给read_result.key和value。
# #
# #     record_bytes = tf.decode_raw(value, tf.uint8)
# #     # 使用tf.decode_raw函数将读取到的原始字节值（value）解码为tf.uint8类型的张量。
# #
# #     read_result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
# #     # 使用tf.strided_slice从record_bytes中提取标签字节，并使用tf.cast将其转换为tf.int32类型，然后赋值给read_result.label。
# #
# #     depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
# #                              [read_result.depth, read_result.high, read_result.weight])
# #     # 从record_bytes中提取图像数据，然后使用tf.reshape将其重新整形为[depth, height, width]的形状，并赋值给depth_major。
# #
# #     read_result.uint8image = tf.transpose(depth_major, [1, 2, 0])
# #     # 使用tf.transpose将depth_major的形状从[depth, height, width]转换为[height, width, depth]，并赋值给read_result.uint8image。
# #
# #     return read_result
# #     # 返回填充了数据的read_result对象。
#
#
# def inputs(data_dir, batch_size, distorted):
#     # 路径拼接,filenames是一个一维数组，包含5个字符串类型的文件名
#     filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
#     # 用这个字符串数组构建一个文件队列
#     file_queue = tf.train.string_input_producer(filenames)
#     read_input = read_cifar10(file_queue)
#     reshaped_image = tf.cast(read_input.uint8image, tf.float32)
#     num_examples_per_epoch = num_examples_pre_epoch_for_train
#     if distorted != None:
#         cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
#         flipped_image = tf.image.random_flip_left_right(cropped_image)
#         adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
#         adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
#         float_image = tf.image.per_image_standardization(adjusted_contrast)
#         float_image.set_shape([24, 24, 3])
#         read_input.label.set_shape([1])
#
#
# inputs("home", 30, True)


# ////////////////////////////////////////////////////
# 第一次
num_classes = 10
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


class CIFAR10Record(object):
    pass


# def read_cifar10(file_queue):
#     result = CIFAR10Record()
#     label_bytes = 1
#     result.height = 32
#     result.width = 32
#     result.depth = 3
#     image_bytes = result.height * result.width * result.depth
#     record_bytes = label_bytes + image_bytes
#     reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#     result.key, value = reader.read(file_queue)
#     image = tf.decode_raw(value, out_type=tf.uint8)
#     result.label = tf.cast(tf.strided_slice(image, [0], [label_bytes]), dtype=tf.int32)
#     depth_major = tf.reshape(tf.strided_slice(image, [label_bytes], [label_bytes + image_bytes])
#                              , [result.depth, result.height, result.width])
#     result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#     return result
#     pass

def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1
    result.high = 32
    result.weight = 32
    result.depth = 3
    image_bytes = result.high * result.weight * result.depth
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes)
    result.key, value = reader.read(file_queue)
    # 字符串解码成张量
    image = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(image, begin=[0], end=[label_bytes]), dtype=tf.int32)
    depth_major = tf.reshape(tf.strided_slice(image, begin=[label_bytes],
                                              end=[label_bytes + image_bytes]),
                             shape=[result.depth, result.high, result.weight])
    # 这里再测试一下
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


# def inputs(data_dir, batch_size, distorted):
#     file_names = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
#     file_queue = tf.train.string_input_producer(file_names)
#     read_input = read_cifar10(file_queue)
#     reshaped_image = tf.cast(read_input.uint8image, dtype=tf.float32)
#     num_examples_pre_epoch = num_examples_pre_epoch_for_train
#     if distorted != None:
#         # cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
#         # flipped_image = tf.image.random_flip_left_right(cropped_image)
#         # brightness_image = tf.image.random_brightness(flipped_image, max_delta=0.8)
#         # contrast_image = tf.image.random_contrast(brightness_image, lower=0.2, upper=1.8)
#         # standard_image = tf.image.per_image_standardization(contrast_image)
#         # standard_image.set_shape([24, 24, 3])
#         # read_input.label.set_shape([1])
#         # min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
#         # print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
#         #       % min_queue_examples)
#         # images_train, label_train = tf.train.shuffle_batch([standard_image, read_input.label]
#         #                                                    , batch_size=batch_size
#         #                                                    , num_threads=16
#         #                                                    , capacity=min_queue_examples + 3 * batch_size
#         #                                                    , min_after_dequeue=min_queue_examples)
#         # return images_train, tf.reshape(label_train, [batch_size])
#
#         #
#         cropped_image = tf.random_crop(reshaped_image, size=[24, 24, 3])
#         flipped_image = tf.image.random_flip_left_right(cropped_image)
#         brightness_image = tf.image.random_brightness(flipped_image, max_delta=0.8)
#         contrast_image = tf.image.random_contrast(brightness_image, lower=0.2, upper=1.8)
#         standard_image = tf.image.per_image_standardization(contrast_image)
#         standard_image.set_shape([24, 24, 3])
#         read_input.label.set_shape([1])
#         min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
#         print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
#               % min_queue_examples)
#         images_train, label_train = tf.train.shuffle_batch([standard_image, read_input.label], batch_size=batch_size,
#                                                            capacity=min_queue_examples + 3 * batch_size,
#                                                            num_threads=16, min_after_dequeue=min_queue_examples
#                                                            )
#         return images_train, tf.reshape(label_train, [batch_size])
#         #
#     else:
#         resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
#         standard_image = tf.image.per_image_standardization(resized_image)
#         standard_image.set_shape([24, 24, 3])
#         read_input.label.set_shape([1])
#         min_queue_examples = int(num_examples_pre_epoch * 0.4)
#         images_train, label_train = tf.train.batch([standard_image, read_input.label], batch_size=batch_size,
#                                                    num_threads=16
#                                                    , capacity=min_queue_examples + 3 * batch_size)
#         return images_train, tf.reshape(label_train, [batch_size])
#
#         pass

def inputs(data_dir, batch_size, distorted):
    file_names = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    file_queue = tf.train.string_input_producer(file_names)
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, dtype=tf.float32)
    num_examples_pre_epoch = num_examples_pre_epoch_for_train
    if distorted != None:
        #
        cropped_image = tf.random_crop(reshaped_image, size=[24, 24, 3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        brightness_image = tf.image.random_brightness(flipped_image, max_delta=0.8)
        contrast_image = tf.image.random_contrast(brightness_image, lower=0.2, upper=1.8)
        standard_image = tf.image.per_image_standardization(contrast_image)
        standard_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)
        images_train, label_train = tf.train.shuffle_batch([standard_image, read_input.label], batch_size=batch_size,
                                                           capacity=min_queue_examples + 3 * batch_size,
                                                           num_threads=16, min_after_dequeue=min_queue_examples
                                                           )
        return images_train, tf.reshape(label_train, [batch_size])
        #
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        standard_image = tf.image.per_image_standardization(resized_image)
        standard_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch * 0.4)
        images_train, label_train = tf.train.batch([standard_image, read_input.label], batch_size=batch_size,
                                                   num_threads=16
                                                   , capacity=min_queue_examples + 3 * batch_size)
        return images_train, tf.reshape(label_train, [batch_size])

        pass
