import matplotlib.image as mping
import numpy as np
import  cv2
import tensorflow as tf

def load_image(path):

    img = mping.imread(path)

    # 将图像修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0]-short_edge)/2)
    xx = int((img.shape[1]-short_edge)/2)
    clip_img = img[yy: yy + short_edge,xx:xx+short_edge]
    return clip_img
def resize_img(image,size):
    with tf.name_scope('resize_image'):
        images = []
        # img 中有多个图片 把每个图片都resize一下
        for i in image:
            i = cv2.resize(i,size)
            images.append(i)
        images = np.array(images)
    return  images

def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
        # 从文件中取出相应的中文，这里我们可以根据需要，比如我们有100个分类，或者
        #有一千个分类可以扩展
        synset = [ l.split(';')[1][:-1] for l in f.readlines()]

    print(synset[argmax])
    return synset[argmax]





