import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

#载入图片+调整图片的尺寸/截取图片的片段
def load_img(path):
    img = mpimg.imread(path)
    short_edge = min(img.shape[:2]) #选择最短的尺寸
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    #剪裁为以中心为准的正方形
    crop_img = img[yy:yy + short_edge,xx:xx + short_edge]
    return crop_img

# 改变图片的大小
def resize_img(img,size):
    with tf.name_scope('resize_img'):
        imgs=[]
        for i in img:
            i = cv2.resize(i,size) #调整为指定尺寸
            imgs.append(i)
        imgs = np.array(imgs)
        return imgs

def print_anwser(argmax):
    # with open("filename", "mode") as f:：上下文管理器，自动管理文件的打开和关闭，确保文件在操作完成后被正确关闭。
    with open("./data/model/index_word.txt",'r',encoding='utf-8') as f:
        #拆分成一个包含标签的列表
        synset = [l.split(";")[1][1] for l in f.readlines()]

    print(synset[argmax])
    return synset[argmax]

