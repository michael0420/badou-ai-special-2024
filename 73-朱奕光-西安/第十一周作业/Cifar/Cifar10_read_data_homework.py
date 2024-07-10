import tensorflow as tf
import os

train_num = 50000
test_num = 10000

class CIFAR10Record(object):
    pass

def read_cifar10(data):
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.channels = 3

    image_bytes = result.height * result.width * result.channels
    record_bytes = image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(data)
    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]), tf.int32)
    img_data = tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes])
    img_data = tf.reshape(img_data, [result.channels, result.height, result.width])
    result.unit8image = tf.transpose(img_data, [1, 2, 0])
    return result

def input_data(distorted, batch_size, data_dir):
    filename = [os.path.join(data_dir, "data_batch_%d.bin" % i)for i in range(1,6)]
    file_queue = tf.train.string_input_producer(filename)
    result = read_cifar10(file_queue)
    data_float32 = tf.cast(result.unit8image, tf.float32)
    if distorted:
        cropped = tf.random_crop(data_float32, [24, 24, 3])
        flipped = tf.image.flip_left_right(cropped)
        brightness = tf.image.random_brightness(flipped, max_delta=0.8)
        contrast = tf.image.random_contrast(brightness, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(contrast)
        image.set_shape([24,24,3])
        result.label.set_shape([1])
        min_examples = int(test_num * 0.4)
        train_image, train_label = tf.train.shuffle_batch(tensors=[image, result.label],
                                                          batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity= min_examples + 3 * batch_size,
                                                          min_after_dequeue = min_examples
                                                          )
        train_label = tf.reshape(train_label, [batch_size])
        return train_image, train_label
    else:
        image = tf.image.resize_image_with_crop_or_pad(data_float32, 24,24)
        image = tf.image.per_image_standardization(image)
        image.set_shape([24,24,3])
        result.label.set_shape([1])
        min_examples = int(test_num * 0.4)
        test_image, test_label = tf.train.batch(tensors=[image, result.label],
                                                batch_size=batch_size,
                                                num_threads=16,
                                                capacity= min_examples + 3 * batch_size)
        test_label = tf.reshape(test_label, [batch_size])
        return test_image, test_label


if __name__ == '__main__':
    train_images, train_labels = input_data(distorted=True, batch_size=100)
    print(train_images)
    print(train_labels)