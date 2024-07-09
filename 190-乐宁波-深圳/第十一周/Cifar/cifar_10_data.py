import os
import tensorflow as tf

num_classes = 10

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


# 定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(filenames):
    label_bytes = 1  # 如果是Cifar-100数据集，则此处为2
    height = 32
    width = 32
    depth = 3  # 因为是RGB三通道，所以深度是3

    image_bytes = height * width * depth  # 图片样本总元素数量
    record_bytes = label_bytes + image_bytes  # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    def _parse_function(proto):
        record_bytes = tf.io.decode_raw(proto, tf.uint8)

        label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
            [depth, height, width]
        )

        uint8image = tf.transpose(depth_major, [1, 2, 0])

        return uint8image, label

    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.map(_parse_function)

    return dataset


def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]  # 拼接地址

    dataset = read_cifar10(filenames)  # 读取数据集

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32)  # 将已经转换好的图片数据再次转换为float32的形式

        if distorted:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
            image = tf.image.random_crop(image, [32, 32, 3])  # 首先将预处理好的图片进行剪切
            image = tf.image.random_flip_left_right(image)  # 将剪切好的图片进行左右翻转
            image = tf.image.random_brightness(image, max_delta=0.8)  # 将左右翻转好的图片进行随机亮度调整
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)  # 将亮度调整好的图片进行随机对比度调整
        else:  # 不对图像数据进行数据增强处理
            image = tf.image.resize_with_crop_or_pad(image, 32, 32)  # 对图片数据进行剪切

        image = tf.image.per_image_standardization(image)  # 进行标准化图片操作
        image.set_shape([32, 32, 3])
        label.set_shape([1])

        return image, tf.reshape(label, [])

    dataset = dataset.map(_preprocess)

    if distorted:
        dataset = dataset.shuffle(buffer_size=int(num_examples_pre_epoch_for_eval * 0.4))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# # 示例使用方法
data_dir = 'cifar_data/cifar-10-batches-bin'
batch_size = 64
distorted = True

train_dataset = inputs(data_dir, batch_size, distorted)
print(train_dataset)
for images, labels in train_dataset.take(1):
    print(images.shape, labels.shape)
