# _*_ coding: UTF-8 _*_
# @Time: 2024/7/2 18:56
# @Author: iris
# @Email: liuhw0225@126.com
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_iamge(img_path):
    # 读取图片信息（rgb）
    img = mpimg.imread(img_path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy::yy + short_edge, xx:xx + short_edge]
    return crop_img


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
    return images


def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]
