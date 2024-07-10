import tensorflow as tf
from tensorflow_core import strided_slice, reshape, transpose
from tensorflow_core.python import decode_raw


class CiFar10Record(object):
    pass


def read_cifar_files(file_queue):
    result = CiFar10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    record_bytes = decode_raw(value, tf.uint8)

    result.label = tf.cast(strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = reshape(strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                          [result.depth, result.height, result.width])

    result.unit8image = transpose(depth_major, [1, 2, 0])
    return result


def inputs(file_lists, batch_size, enhance, request_num):
    file_queue = tf.train.string_input_producer(file_lists)
    ci_far_record = read_cifar_files(file_queue)
    reshaped_image = tf.cast(ci_far_record.unit8image, tf.float32)

    if enhance:
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        float_image = tf.image.per_image_standardization(adjusted_contrast)
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([24, 24, 3])
    ci_far_record.label.set_shape([1])

    min_queue_examples = int(request_num * 0.4)
    images_result, labels_result = tf.train.shuffle_batch([float_image, ci_far_record.label], batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue=min_queue_examples,
                                                          )
    return images_result, reshape(labels_result, [batch_size])
