import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

# img=load_image("Test.jpg")
# cv2.imshow("img",img)
# cv2.waitKey(0)

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


# def load_image(path):
#     img=mpimg.imread(path)
#     # 将图片修剪成中心正方形
#     short_edge=min(img.shape[0],img.shape[1])
#     yy=(img.shape[0]-short_edge)//2
#     xx=(img.shape[1]-short_edge)//2
#     crop_img=img[yy:yy+short_edge,xx:xx+short_edge]
#     return crop_img
#
# def resize_img(images,new_size):
#     ret=[]
#     for img in images:
#         img=cv2.resize(img,new_size)
#         ret.append(img)
#     ret=np.array(ret)
#     return ret
#
# def print_answer(argmax):
#     with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
#         hash=[l.split(';')[1][:-1] for l in f.readlines()]
#     return hash[argmax]
#     pass
#
#
# # 第二次
# def load_image(path):
#     img=cv2.imread(path)
#     short_edge=min(img.shape[0],img.shape[1])
#     yy=(img.shape[0]-short_edge)//2
#     xx=(img.shape[1]-short_edge)//2
#     crop_img=img[yy:yy+short_edge,xx:xx+short_edge]
#     return crop_img
#
# def resize_img(images,new_size):
#     ret=[]
#     for img in images:
#         tmp=cv2.resize(img,new_size)
#         ret.append(tmp)
#     ret=np.array(ret)
#     return ret
#
# def print_answer(argmax):
#     with open('./xxxxxxxxxx','r') as f:
#        hash=[l.split(';')[1][:-1] for l in f.readlines()]
#     return hash[argmax]