import os

import cv2
import numpy as np
import tensorflow as tf


def get_files(bash_bath, synset="./synset.txt"):
    image_list = []
    label_list = []
    # os.remove(synset)
    # synset_file = open(synset, 'x')
    # 获取所有分类
    classes = [x for x in os.listdir(bash_bath) if os.path.isdir(bash_bath)]
    # 遍历所有分类
    for index, name in enumerate(classes):
        class_path = os.path.join(bash_bath, name)
        # 遍历单个分类
        lines = "%d:%s\n" % (index, name)
        # synset_file.write(lines)
        for image_name in os.listdir(class_path):
            image_list.append(os.path.join(class_path, image_name))
            label_list.append(index)
    # synset_file.close()

    # 将文件和标签打乱后保存
    images = []
    labels = []
    indices = list(range(len(image_list)))
    np.random.shuffle(indices)
    for i in indices:
        images.append(image_list[i])
        labels.append(label_list[i])
    np.random.shuffle([images, labels])
    return images, labels


def write_tf_record(base_path, tfrecord_path='./', train_data=True):
    if not os.path.isdir(tfrecord_path):
        os.mkdir(tfrecord_path)
    image_list, label_list = get_files(base_path)
    # 每个TFRecord文件的样本长度
    per_record_length = 10000
    num_record = int(np.ceil(len(image_list) / per_record_length))
    print("TFRecord文件数：%d" % num_record)
    for i in range(num_record):
        if train_data:
            file_name = os.path.join(tfrecord_path, "train_data-tfrecord-%d-of-%d.bin" % (i + 1, num_record))
        else:
            file_name = os.path.join(tfrecord_path, "test_data-tfrecord-%d-of-%d.bin" % (i + 1, num_record))
        file_writer = tf.io.TFRecordWriter(file_name)
        start_index = i * per_record_length
        end_index = np.min([(i + 1) * per_record_length, len(image_list)])

        # 遍历并读取每一组样本
        for image_path, label in zip(image_list[start_index: end_index], label_list[start_index:end_index]):
            img = cv2.imread(image_path)
            # 将数据转换为二进制数据
            raw_img = img.tobytes()
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_img])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            file_writer.write(example.SerializeToString())
        file_writer.close()


def read_and_decode(file_name_list, img_shape, epochs=None):
    file_queue = tf.train.string_input_producer(file_name_list)
    tf_reader = tf.TFRecordReader()
    _, serialized_example = tf_reader.read(file_queue)
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(serialized_example, features=features)
    img = tf.reshape(tf.decode_raw(sample['image'], tf.uint8), img_shape)
    label = tf.reshape(tf.cast(sample['label'], tf.int32), [1])
    return img, label


def input_data(filenames, batch_size, capacity=4096, min_after_dequeue=1024, num_threads=10):
    img, label = read_and_decode(file_name_list=filenames, img_shape=[299, 299, 3])
    images_batch, labels_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    return images_batch, labels_batch


def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    return np.resize(image, size, method, align_corners)


def enforce_train_data(source_path, des_path="./train_data", shape=(299, 299)):
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    # 获取所有分类
    classes = [x for x in os.listdir(source_path) if os.path.isdir(source_path)]
    # 遍历所有分类
    for index, name in enumerate(classes):
        dest_class_path = os.path.join(des_path, str(index))
        if not os.path.exists(dest_class_path):
            os.mkdir(dest_class_path)
        class_path = os.path.join(source_path, name)
        # 遍历单个分类
        for image_name in os.listdir(class_path):
            file_name = os.path.join(class_path, image_name)
            dest_filename = os.path.join(dest_class_path, image_name)
            img = cv2.imread(file_name)
            img = cv2.resize(img, shape, interpolation=tf.image.ResizeMethod.BILINEAR)
            cv2.imwrite(dest_filename, img)


def get_train_dataset(bash_bath):
    image_list = []
    label_list = []
    # 获取所有分类
    classes = [x for x in os.listdir(bash_bath) if os.path.isdir(bash_bath)]
    # 遍历所有分类
    for index, name in enumerate(classes):
        class_path = os.path.join(bash_bath, name)
        # 遍历单个分类
        lines = "%d:%s\n" % (index, name)
        # synset_file.write(lines)
        for image_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, image_name))
            image_list.append(img)
            label_list.append(index)

    # 将文件和标签打乱后保存
    images = []
    labels = []
    indices = list(range(len(image_list)))
    np.random.shuffle(indices)
    for i in indices:
        images.append(image_list[i])
        labels.append(label_list[i])
    np.random.shuffle([images, labels])
    return np.reshape(images, [9808, 299, 299, 3]), np.reshape(labels, [9808, 1])


if __name__ == '__main__':
    # write_tf_record("./train_data")
    # enforce_train_data("./train_set")
    files = ['train_data-tfrecord-1-of-1.bin']
    train_data, train_label = input_data(filenames=files, batch_size=200, capacity=4096, min_after_dequeue=1024,
                                         num_threads=5)
    image_train, label_train = read_and_decode(file_name_list=files, img_shape=[299, 299, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        result = sess.run([train_data, train_label])
        print(result[0])
        # result = sess.run([train_data, train_label])
        # result = sess.run([image_train, label_train])
        # cv2.imshow(str(result[1]), result[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        coord.request_stop()
        coord.join(threads)
