import matplotlib.image as mpimg
import numpy as np
import cv2
from tensorflow.python.ops import array_ops
import tensorflow as tf

def import_image(path):
    # rgb
    img = mpimg.imread(path)
    # 修成正方形
    print(img.shape)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy+short_edge, xx: xx+short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

# 打印分类的结果
def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding="utf-8") as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]
