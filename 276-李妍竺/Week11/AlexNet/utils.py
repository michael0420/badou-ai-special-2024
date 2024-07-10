import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops #类似np.array

def load_image(path):
    # 读取图片，并将图片修剪成正方形
    img = mpimg.imread(path) #rgb
    short_edge = min(img.shape[:2])
    h = int((img.shape[0] - short_edge) / 2)
    w = int((img.shape[1] - short_edge) / 2)
    crop_img = img[h:h+short_edge,w:w+short_edge]
    return crop_img

def resize_image(image,size):  #这里的size是 h w
    with tf.name_scope('resize_image'): #tf.name_scope 给变量包一层名字 'resize_image' 是提供的范围名称。它用于为变量或操作命名，以帮助识别和理解代码。
        images = []
        for i in image:
            i = cv2.resize(i,size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:   #因为有中文，所以要用utf-8
        synset = [l.split(';')[1][:-1] for l in f.readlines()]  #[:-1] 去掉末尾字符，比如换行符 空格之类的

    print(synset)
    print(synset[argmax])
    return synset[argmax]

#print_answer(0)



