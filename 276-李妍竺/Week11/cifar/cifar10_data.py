#数据预处理
import os   #目录相关
import tensorflow as tf

#总分类数
num_classes = 10

#用于训练与评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

#定义一个空类，用于返回读取的Cifar-10的数据
class Cifar10Record(object):
    pass

#定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    result=Cifar10Record()

    label_bytes=1                                            #如果是Cifar-100数据集，则此处为2
    result.height=32
    result.width=32
    result.depth=3                                           #因为是RGB三通道，所以深度是3

    image_bytes = result.height * result.width * result.depth  # 图片样本总元素数量
    record_bytes = label_bytes + image_bytes  # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    reader = tf.FixedLengthRecordReader(
        record_bytes=record_bytes)  # 使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    result.key, value = reader.read(file_queue)  # 使用该类的read()函数从文件队列里面读取文件

    record_bytes = tf.decode_raw(value, tf.uint8)  # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组

    # 因为该数组第一个元素是标签，所以用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    '''
    tf.strided_slice(输入数据，开始切片处，终止切片处，步长...)
    '''
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    # 将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])  # 换顺序

    return result  # 返回值是已经把目标文件里面的信息都读取出来

def inputs(data_dir,batch_size,distorted):               #这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    '''
    os.path.join(data_dir,"a"%): data_dir 和 a 拼接
    '''
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]  #拼接地址(一共5个)

    file_queue=tf.train.string_input_producer(filenames) #根据已经有的文件地址创建一个文件队列
    read_input=read_cifar10(file_queue)                      #根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image=tf.cast(read_input.uint8image,tf.float32) #将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch=num_examples_pre_epoch_for_train

    if distorted != None:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理

        cropped_image=tf.random_crop(reshaped_image,[24,24,3]) #首先将预处理好的图片进行剪切，使用tf.random_crop()函数

        flipped_image=tf.image.random_flip_left_right(cropped_image)    #将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.85)   #将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.6)    #将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数

        float_image=tf.image.per_image_standardization(adjusted_contrast)          #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24,24,3])                      #设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)  # 4000
        print("Filling queue with %d CIFAR images before starting to train..."% min_queue_examples)

        '''
                num_threads:使用多线程读取文件
                min_after_dequeue:当一次出列操作完成后，队列中元素的最小数量
                '''
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train, tf.reshape(labels_train, [batch_size])

    else:  # 不对图像数据进行数据增强处理
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24,
                                                               24)  # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])




