import os
import tensorflow as tf

num_class = 10
# 训练数
for_train = 50000
# 样本shu
for_eval = 10000


# 定义一个空类,用于返回
class MYCIFAR10read(object):
    pass


# 读取目标文件里面的内容
def read_file_cifar10(file_cifar):
    result = MYCIFAR10read()
    lable_bytes = 1  # 如果是Cifar-100数据集，则此处为2
    result.high = 32  # H
    result.width = 32  # W
    result.depth = 3  # C
    # 图片样本总元素数量
    image_bytes = result.high * result.width * result.depth
    assemble_bytes = lable_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=assemble_bytes)
    # 读取文件队列里的文件
    read_k, read_v = reader.read(file_cifar)
    # 将读取到的文件内容从字符串形式解析为图像对应的像素数组
    assemble_bytes = tf.decode_raw(read_v, tf.uint8)
    # 数组第一个元素是标签
    result.lable = tf.cast(tf.strided_slice(assemble_bytes, [0], [lable_bytes]), tf.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    chw_maj = tf.reshape(tf.strided_slice(assemble_bytes, [lable_bytes], [lable_bytes + image_bytes]),
                         [result.depth, result.high, result.width])
    result.uint8image = tf.transpose(chw_maj, [1, 2, 0])
    return result  # 返回值是已经把目标文件里面的信息都读取出来


# 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def Preprocessing_data(data_dir, batch_size, distorted):
    # 地址拼接
    url_s = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    # 根据已经有的文件地址创建一个文件队列
    file_producer = tf.train.string_input_producer(url_s)
    cifar_input = read_file_cifar10(file_producer)
    # 将已经转换好的图片数据再次转换为float32的形式
    reashp_imade = tf.cast(cifar_input.uint8image, tf.float32)
    num_examples_per_epoch = for_train
    if distorted != None:  # 进行图片增强处理
        # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        crop_image = tf.random_crop(reashp_imade, [24, 24, 3])
        # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
        overturn_image = tf.image.random_flip_left_right(crop_image)
        # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        brightness_image = tf.image.random_brightness(overturn_image, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        contrast_image = tf.image.random_contrast(brightness_image, lower=0.2, upper=1.8)
        # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        standardization_image = tf.image.per_image_standardization(contrast_image)
        standardization_image.set_shape([24, 24, 3])
        cifar_input.lable.set_shape([1])

        min_queue_examples = int(for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        images_train, images_labels = tf.train.shuffle_batch([standardization_image, cifar_input.lable],
                                                             batch_size=batch_size,
                                                             num_threads=16,
                                                             capacity=min_queue_examples + 3 * batch_size,
                                                             min_after_dequeue=min_queue_examples, )
        return images_train, tf.reshape(images_labels, [batch_size])
    else:
        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resize_image = tf.image.resize_image_with_crop_or_pad(reashp_imade, 24, 24)
        # 剪切完成以后，直接进行图片标准化操作
        standardization = tf.image.per_image_standardization(resize_image)
        standardization.set_shape([24, 24, 3])
        cifar_input.lable.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch * 0.4)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        images_train, images_labels = tf.train.batch([standardization, cifar_input.lable], batch_size=batch_size,
                                                     num_threads=16,
                                                     capacity=min_queue_examples + 3 * batch_size)

        return images_train, tf.reshape(images_labels, [batch_size])