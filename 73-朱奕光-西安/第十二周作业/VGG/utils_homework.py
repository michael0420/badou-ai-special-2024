import cv2
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

def read_image(filename):
    img = mpimg.imread(filename)
    h, w, c = img.shape
    short_edge = min(h,w)
    h_s = int((h - short_edge)/2)
    w_s = int((w - short_edge)/2)
    img = img[h_s:h_s+short_edge, w_s:w_s+short_edge]
    return img
    # print(img.shape)

def resize_image(img, new_size):
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_images(img, new_size)
    img = tf.reshape(img, [-1, new_size[0], new_size[1], 3])
    return img
    # print(img.shape)

def print_top(res, filepath):
    labels = [l.strip() for l in open(filepath).readlines()]
    pred = np.argsort(res)[::-1]
    top1 = labels[pred[0]]
    print('top1:', top1, res[pred[0]])
    top5 = [(labels[pred[i]], res[pred[i]])for i in range(5)]
    print('top5:', top5)

