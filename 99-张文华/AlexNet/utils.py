'''
工具包
'''
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images


# 定义一个方法打印输出标签索引对应的类别
def print_answer(argmax):
    with open('./data/model/index_word.txt', 'r', encoding='utf-8') as f:
        index_set = [i.split(';')[1] for i in f.readlines()]
        # print(index_set[argmax])
        return index_set[argmax]


