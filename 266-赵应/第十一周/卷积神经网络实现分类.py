import os, time
import tensorflow as tf


def read_image(filenames):
    file_queue = tf.train.string_input_producer(filenames)
    label_bytes = 1
    image_width = 32
    image_height = 32
    image_depth = 3
    images_bytes = image_depth * image_height * image_width
    record_bytes = label_bytes + images_bytes
    file_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    filename, raw_data = file_reader.read(file_queue)
    source_data = tf.decode_raw(raw_data, tf.uint8)
    label = tf.cast(tf.strided_slice(source_data, [0], [label_bytes]), tf.int32)
    image = tf.reshape(tf.strided_slice(source_data, [label_bytes], [record_bytes]),
                       [image_depth, image_height, image_width])
    return [image, label]


def image_enhance(image, label, batch_size, distorted=False):
    train_image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
    label.set_shape([1])
    if distorted:
        # 剪裁图片到指定大小
        cropped_image = tf.image.random_crop(train_image, [24, 24, 3])
        # 左右翻转图片
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 调整图片亮度
        adjust_brightness = tf.image.random_brightness(flipped_image, max_delta=.8)
        # 调整图片对比度
        adjust_saturation = tf.image.random_saturation(adjust_brightness, lower=.2, upper=1.8)
        # 图片标准化
        standard_image = tf.image.per_image_standardization(adjust_saturation)
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        images, labels = tf.train.shuffle_batch([standard_image, label], batch_size=batch_size,
                                               num_threads=16, capacity=min_queue_examples + 3 * batch_size,
                                               min_after_dequeue=min_queue_examples)
    else:
        resize_image = tf.image.resize_image_with_crop_or_pad(train_image, 24, 24)
        standard_image = tf.image.per_image_standardization(resize_image)
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        standard_image.set_shape([24, 24, 3])
        images, labels = tf.train.batch([standard_image, label], batch_size=batch_size, num_threads=16,
                                                    capacity= min_queue_examples + 3* batch_size)
    return [images, tf.reshape(labels, [batch_size])]


def variable_with_weight_loss(shape, stddev, w):
    # 根据随机分布参数stddev,卷积核shape生成卷积核
    kernel = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if w is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(kernel), w, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return kernel


if __name__ == '__main__':
    num_examples_pre_epoch_for_train = 50000
    num_examples_pre_epoch_for_eval = 10000
    batch_size = 100
    epochs = 100
    # 训练数据放到当前位置的cifar-10-batches-bin目录下
    files = [os.path.join("cifar-10-batches-bin", "data_batch_%d.bin"%i) for i in range(1, 6)]
    images, labels = read_image(files)
    train_images, train_labels = image_enhance(images, labels, batch_size=batch_size, distorted=True)

    # 构建神经网络
    train_data = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    train_label = tf.placeholder(tf.int32, [batch_size])
    # 第一层神经网络
    l1_kernel = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, w=0.0)
    l1_conv = tf.nn.conv2d(train_data, l1_kernel, [1, 1, 1, 1], padding="SAME")
    l1_bias = tf.constant(.0, shape=[64])
    l1_relu = tf.nn.relu(tf.nn.bias_add(l1_conv, l1_bias))
    l1_pool = tf.nn.max_pool(l1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第二层卷积神经网络
    l2_kernel = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, w=.0)
    l2_conv = tf.nn.conv2d(l1_pool, l2_kernel, [1, 1, 1, 1], padding="SAME")
    l2_bias = tf.Variable(tf.constant(0.1, shape=[64]))
    l2_relu = tf.nn.relu(tf.nn.bias_add(l2_conv, l2_bias))
    l2_pool = tf.nn.max_pool(l2_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 重新构造pool2形状，便于后续做全连接, 第一位为batch_size代表第一个维度大小为batch_size，第二维度为-1代表将原来数据除第一维以外的维度压缩为1维
    reshape_data = tf.reshape(l2_pool, [batch_size, -1])
    dim2 = reshape_data.get_shape()[1].value

    # 第三层为全连接层
    l3_weight = variable_with_weight_loss([dim2, 384], stddev=.04, w=.004)
    l3_bias = tf.constant(.1, shape=[384])
    l3_output = tf.nn.relu(tf.matmul(reshape_data, l3_weight) + l3_bias)

    # 第四层为全连接层
    l4_weight = variable_with_weight_loss([384, 192], stddev=.04, w=.004)
    l4_bias = tf.constant(.1, shape=[192])
    l4_output = tf.nn.relu(tf.matmul(l3_output, l4_weight) + l4_bias)

    # 第五层为全连接层,也是输出层
    l5_weight = variable_with_weight_loss([192, 10], stddev=1 / 192.0, w=.0)
    l5_bias = tf.constant(.1, shape=[10])
    result = tf.add(tf.matmul(l4_output, l5_weight), l5_bias)

    # 计算损失函数 = 稀疏多分类交叉熵损失函数 + 参数正则化损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(train_label, tf.int64))
    weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
    loss_function = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
    # 优化函数
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss_function)
    # 计算top k准确率
    top_k = tf.nn.in_top_k(result, tf.cast(train_label, tf.int64), 1)

    # 初始化全局参数
    init_op = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        sess.run(init_op)
        for i in range(epochs):
            step = i + 1
            start_time = time.time()
            train_batch_data, train_batch_label = sess.run([train_images, train_labels])
            _, loss = sess.run([train_op, loss_function], feed_dict={train_data: train_batch_data, train_label: train_batch_label})
            duration = time.time() - start_time
            print("step: ", step)
            if step % 100 == 0:
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
                    step, loss, examples_per_sec, sec_per_batch))
        coord.request_stop()
        coord.join(threads)
        tf.nn.conv3d



