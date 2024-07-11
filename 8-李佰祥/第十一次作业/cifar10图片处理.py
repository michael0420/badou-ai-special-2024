import os
import tensorflow as tf

num_classes=10

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000
class CIFAR10Record(object):
    pass



#该函数主要实现读取二进制图片，再解析为uint8格式图片，再把
#图片数据从c,h,w  转换为h,w,c
def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1   #(定义了标签占用的字节数)1代表了是cifar10
    #动态属性设置，这是Python语言的一个特性。在Python中，你可以动态地给一个对象添加属性
    result.height=32
    result.width=32
    result.depth=3   #彩色图。深度为3

    image_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes + image_bytes
    #创建读取器(读取固定长度记录的文件)，所以需要传入每个记录的字节大小record_bytes
    reader = tf.FixedLengthRecordReader(record_bytes)
    #value将包含一个32x32 RGB图像和一个标签的二进制表示
    #key是包含了文件名和记录在文件中的位置信息
    result.key , value = reader.read(file_queue)
    #用于将一个包含原始字节数据的字符串张量解码成一个扁平的一维张量。这个函数特别适用于将从文件中读取的连续字节流转换为数值数据
    record_bytes = tf.decode_raw(value,tf.uint8)
    #拆分标签,标签通常需要作为一个整数类型来处理，以便用于分类任务(故转换为int32)
    #strided_slice用来从record_bytes中提取前label_bytes个字节，这
    # 通常是包含标签信息的部分。[0]和[label_bytes]定义了切片的开始和结束位置
    result.label =tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    #再拆分数据,将一维数据转换为3维
    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),
                  [result.depth,result.height,result.width])

    #转换为高度，宽度和通道结构，tf.transpose用来改变张量的维度顺序
    #tf.transpose 需要一个perm参数，这是一个整数列表，定义了新的维度顺序。
    # 例如，对于一个形状为(3, 4, 5)的张量，
    # tf.transpose(t, [2, 0, 1])将维度顺序变为(5, 3, 4)。
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result



def input(data_dir,batch_size,distorted):
    #拼接地址
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    #根据filename创建文件队列
    #string_input_producer是构建文件输入管线的重要步骤
    #这个队列会被多个读取线程共享，每个线程可以从队列中取出一个文件名，然后读取对应文件的内容
    #这种队列机制是为了实现高效的数据读取和并行处理而设计的。通过使用文件名队列，我们可以并行地从多个文件中读取数据，这对于大规模数据集的处理是非常必要的
    file_queue = tf.train.string_input_producer(filenames)  # 根据已经有的文件地址创建一个文件队列
    read_input = read_cifar10(file_queue)

    #将uint8在转换为float32
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch = num_examples_pre_epoch_for_train

    if distorted!=None:
        #剪切操作
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])
        #对剪切好的图片进行左右翻转
        flipped_image=tf.image.random_flip_left_right(cropped_image)
        #对翻转后的图片进行随机亮度调整
        #如果max_delta设置为0.8，这意味着亮度可以随机增加或减少原始亮度值的80%
        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)
        #随机对比度调整
        #lower=0.2意味着调整后的对比度最低可以是原始对比度的20%
        #upper=1.8意味着调整后的对比度最高可以是原始对比度的180%
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        #图片标准化,tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        float_image=tf.image.per_image_standardization(adjusted_contrast)
        #如果没有显式地使用set_shape，TensorFlow可能仍然将float_image的形状视为部分或完全未知。这可能导致后续操作（如模型中的层）无法确定输入的准确形状
        float_image.set_shape([24,24,3])
        #print(float_image)#设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )

        return images_train,tf.reshape(labels_train,[batch_size])


    else:
        #该函数如果原始图片比目标尺寸大，函数会从中心裁剪出一个24x24的区域，这样就可能移除图片的边缘部分，但保留了中心区域的信息
        #如果原始图片比目标尺寸小，函数会在图片的边缘添加灰度填充（默认情况下），使得图片的尺寸达到24x24。填充的颜色通常是0，即黑色，但这可以通过传递额外的参数来改变
        #如果原始图片的尺寸正好是24x24，则不需要进行任何操作，图片将直接被返回
        #能够确保输出图片尺寸一致，同时尽可能地保留图片的原始内容
        #这在机器学习项目中非常有用，因为许多模型要求输入的图片具有相同的尺寸
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image=tf.image.per_image_standardization(resized_image)          #剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test,tf.reshape(labels_test,[batch_size])



















if __name__ == '__main__':
    input('cifar_data/cifar-10-batches-bin',100,True)