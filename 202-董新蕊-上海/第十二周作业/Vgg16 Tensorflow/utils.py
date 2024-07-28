#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
import matplotlib.image as mping


# In[4]:


def load_image(path):
    img = mping.imread(path) #import matplotlib.image as mpimg mpimg.imread读出来的值为0-1 from PIL import Image Image.open读出来的值为0-255
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge)/2)
    xx = int((img.shape[1] - short_edge)/2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    return crop_img


# In[5]:


def resized_img(image, resized_shape,method=tf.image.ResizeMethod.BILINEAR,align_corners=False): #方法为双线性差值
    #首先增加图像维度（batchsize），开始resize，之后进行reshape
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image,0)
        image = tf.image.resize_images(image, resized_shape, method, align_corners)
        iamge = tf.reshape(image,[-1, resized_shape[0], resized_shape[1], 3])
        return image


# In[6]:


def print_prob(prob, file_path):
    
    sysnet = [l.strip() for l in open(file_path).readlines()]
    #n01440764 tench, Tinca tinca sysnet文件内容格式
    #用于移除字符串首尾的空白字符（包括空格、制表符 \t、换行符 \n 等）。
    pred = np.argsort(prob)[::-1]
    top1 = sysnet[pred[0]]
    print('top1:', top1, prob[pred[0]])
    top5 = [(sysnet[pred[i]], prob[pred[i]]) for i in range(5)]
    print('top5', top5)
    return top1, top5
    
    


# In[ ]:




