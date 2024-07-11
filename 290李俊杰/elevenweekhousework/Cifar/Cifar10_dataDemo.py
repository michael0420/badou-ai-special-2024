#该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf
num_classes=10

# 设定用于训练和测试的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

# 定义一个空类，用于返回读取的Cifar——10的数据
class CIFAR10Record(object):
    pass

#定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1
    result.height=32
    result.width=32
    result.depth=3

    # 图片样本总元素量
    image_bytes=result.height*result.width*result.depth
    record_bytes=label_bytes+image_bytes

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value=reader.read(file_queue)

    record_bytes=tf.decode_raw(value,tf.uint8)

    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),
                           [result.depth,result.height,result.width])

    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result

def inputs(data_dir,batch_size,distorted):
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]

    file_queue=tf.train.string_input_producer(filenames)
    read_input=read_cifar10(file_queue)
    reshaped_image=tf.cast(read_input.uint8image,tf.float32)

    num_examples_pre_epoch=num_examples_pre_epoch_for_train

    if distorted!=None:
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])

        flipped_image=tf.image.random_flip_left_right(cropped_image)

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)

        float_image=tf.image.per_image_standardization(adjusted_contrast)

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval*0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples+3*batch_size,
                                                         min_after_dequeue=min_queue_examples)

        return  images_train,tf.reshape(labels_train,[batch_size])


    else:
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)

        float_image=tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch*0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                               batch_size=batch_size,num_threads=16,
                                               capacity=min_queue_examples+3*batch_size)

        return images_test,tf.reshape(labels_test,[batch_size])







