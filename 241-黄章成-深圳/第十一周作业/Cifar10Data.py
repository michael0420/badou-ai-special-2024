import os
import tensorflow as tf


class Cifar10DataProcess:
    def __init__(self, data_dir, num_examples_pre_epoch_for_train=50000, num_examples_pre_epoch_for_eval=10000):
        """
        参数初始化
        :param data_dir: 数据所在文件夹目录
        :param num_examples_pre_epoch_for_train: 训练集样本数
        :param num_examples_pre_epoch_for_eval: 测试集样本数
        """
        self.data_dir = data_dir
        self.num_examples_pre_epoch_for_train = num_examples_pre_epoch_for_train
        self.num_examples_pre_epoch_for_eval = num_examples_pre_epoch_for_eval

    def data_read(self):
        """
        定义一个数据读取器
        :return: 返回输入数据和标签
        """
        filenames = [os.path.join(self.data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
        file_queue = tf.train.string_input_producer(filenames)
        label_bytes = 1
        height = 32
        width = 32
        depth = 3
        img_bytes = height * width * depth
        record_bytes = label_bytes + img_bytes
        reader = tf.FixedLengthRecordReader(record_bytes)
        key, value = reader.read(file_queue)
        value = tf.decode_raw(value, tf.uint8)
        labels = tf.cast(tf.strided_slice(value, [0], [label_bytes]), tf.int32)
        imgs = tf.reshape(tf.strided_slice(value, [label_bytes], [record_bytes]),
                          [depth, height, width])
        uint8imgs = tf.transpose(imgs, [1, 2, 0])
        return uint8imgs, labels

    def data_process(self, batch_size, enhance=False):
        """
        数据增强处理
        :param batch_size: 批大小
        :param enhance: 是否需要数据增强，True是需要数据增强，训练集需要数据增强，测试集不用。
        :return: 返回处理好的输入数据和标签数据
        """
        print('data reading...')
        imgs, labels = self.data_read()
        print('data read end...')
        imgs = tf.cast(imgs, tf.float32)
        num_examples_per_epoch = self.num_examples_pre_epoch_for_train
        labels.set_shape([1])
        if enhance is True:
            print('crop...')
            croped_imgs = tf.random_crop(imgs, [24, 24, 3])  # 裁剪
            print('flip...')
            fliped_imgs = tf.image.random_flip_left_right(croped_imgs)  # 翻转
            print('brightness...')
            brightness_imgs = tf.image.random_brightness(fliped_imgs, max_delta=0.8)  # 调整亮度
            print('contrast...')
            contrast_imgs = tf.image.random_contrast(brightness_imgs, lower=0.2, upper=1.8)  # 调整对比度
            print('std...')
            std_imgs = tf.image.per_image_standardization(contrast_imgs)  # 标准化
            std_imgs.set_shape([24, 24, 3])
            min_queue_examples = int(self.num_examples_pre_epoch_for_eval * 0.4)
            print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
                  % min_queue_examples)
            train_imgs, train_labels = tf.train.shuffle_batch([std_imgs, labels], batch_size=batch_size,
                                                              num_threads=16,
                                                              capacity=min_queue_examples + 3 * batch_size,
                                                              min_after_dequeue=min_queue_examples
                                                              )
            return train_imgs, tf.reshape(train_labels, [batch_size])
        else:
            resized_image = tf.image.resize_image_with_crop_or_pad(imgs, 24, 24)
            std_image = tf.image.per_image_standardization(resized_image)
            std_image.set_shape([24, 24, 3])
            min_queue_examples = int(num_examples_per_epoch * 0.4)

            test_imgs, test_labels = tf.train.batch([std_image, labels],
                                                    batch_size=batch_size, num_threads=16,
                                                    capacity=min_queue_examples + 3 * batch_size)
            return test_imgs, test_labels


if __name__ == '__main__':
    data_dir = '.\cifar_data\cifar-10-batches-bin'
    num_examples_pre_epoch_for_train = 50000
    num_examples_pre_epoch_for_eval = 10000
    cifar10 = Cifar10DataProcess(data_dir, num_examples_pre_epoch_for_train, num_examples_pre_epoch_for_eval)
    inputs, labels = cifar10.data_process(batch_size=100, enhance=True)