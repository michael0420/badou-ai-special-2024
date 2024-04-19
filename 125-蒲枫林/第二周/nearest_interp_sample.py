'''
使用最临近插值法实现图像缩放
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

class MyImg:
    img = None
    ch_img = None

    dst_height = None
    dst_width = None

    def __init__(self,path):
        self.img = cv2.imread(path)

    def set_size(self,ds_ht,ds_wd):
        self.dst_height = ds_ht
        self.dst_width = ds_wd

    def show_img(self):
        img_tmp = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,1)
        plt.imshow(img_tmp)

    def cvt_img(self):
        h,w,c = self.img.shape
        ch_img = np.zeros((self.dst_height,self.dst_width,c),np.uint8)

        ch_h = self.dst_height / h
        ch_w = self.dst_width / w

        for i in range(self.dst_height):
            for j in range(self.dst_width):
                h1 = round(i/ch_h)
                w1 = round(j/ch_w)
                ch_img[i,j] = self.img[h1,w1]
        self.ch_img = ch_img

    def show_ch_img(self):
        img_tmp = cv2.cvtColor(self.ch_img,cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,2)
        plt.imshow(img_tmp)


my_img = MyImg('./lenna.png')
my_img.show_img()

my_img.set_size(800,800)
my_img.cvt_img()

my_img.show_ch_img()
plt.show()



