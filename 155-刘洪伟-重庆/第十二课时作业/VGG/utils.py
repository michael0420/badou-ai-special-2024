# _*_ coding: UTF-8 _*_
# @Time: 2024/7/8 20:57
# @Author: iris
# @Email: liuhw0225@126.com
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_image(file_path):
    """
    读取图片信息
    :param file_path: 文件路径
    :return:
    """
    image = plt.imread(file_path)
    # 将图片裁剪成中心的正方形
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    crop_image = image[yy:yy + short_edge, xx: xx + short_edge]
    return crop_image


def resize_iamge(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    """
    resize,reshape 图片
    :param image: 图片信息
    :param size: 大小
    :param method: 双线性插值法
    :param align_corners:
    :return:
    """
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(images=image, size=size, method=method, align_corners=align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


def print_prob(prob, file_path):
    """
    打印信息
    :param prob:
    :param file_path:
    :return:
    """
    synset = [i.strip() for i in open(file=file_path).readlines()]
    # 倒序排序
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1
