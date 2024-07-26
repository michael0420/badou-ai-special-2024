import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

class utils():

    def load_image(path):
        # 读取图片，rgb
        img = mpimg.imread(path)
        # 将图片修剪成中心的正方形
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        return crop_img

    def resize_image(image, size,
                     method=tf.image.ResizeMethod.BILINEAR,
                     align_corners=False):
        with tf.name_scope('resize_image'):
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, size,
                                           method, align_corners)
            image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
            return image

    def print_prob(prob, file_path):
        synset = [l.strip() for l in open(file_path).readlines()]
        # 将概率从大到小排列的结果的序号存入pred
        pred = np.argsort(prob)[::-1]
        # 取最大的1个、5个。
        top1 = synset[pred[0]]
        print(("Top1: ", top1, prob[pred[0]]))
        top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
        print(("Top5: ", top5))
        return top1


slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

if __name__ == '__main__':
    img1 = utils.load_image("./test_data/table.jpg")

    # 对输入的图片进行resize，使其shape满足(-1,224,224,3)
    inputs = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = utils.resize_image(inputs, (224, 224))

    # 建立网络结构
    prediction = vgg_16(resized_img)

    # 载入模型
    sess = tf.Session()
    ckpt_filename = './model/vgg_16.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    # 最后结果进行softmax预测
    pro = tf.nn.softmax(prediction)
    pre = sess.run(pro, feed_dict={inputs: img1})

    # 打印预测结果
    print("result: ")
    utils.print_prob(pre[0], './synset.txt')