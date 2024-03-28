# -*- coding: utf-8 -*-

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

config_value = "img_one"            #图像处理算法方式
img = cv2.imread("sj.jpg")

# 图像灰度化
def gray(img):
    if config_value == "img_one":
        h,w = img.shape[:2]                               #获取图片的high和wide
        img_gray = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
        for i in range(h):
            for j in range(w):
                m = img[i,j]                             #取出当前high和wide中的BGR坐标
                img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
    elif config_value == "img_two":
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


# 图像黑白化（二值法）
def black_white(img):
    img_gray_two = rgb2gray(img)
    rows, cols = img_gray_two.shape
    for i in range(rows):
        for j in range(cols):
         if (img_gray_two[i, j] <= 0.5):
            img_gray_two[i, j] = 0
        else:
            img_gray_two[i, j] = 1
    img_binary = np.where(img_gray_two >= 0.5, 1, 0)
    return img_binary


# 图像放大缩小(临近插值）
def resize(img,high,wide):
    if config_value == "img_one":
        height, width, channels = img.shape
        emptyImage = np.zeros((high, wide, channels), np.uint8)
        sh = high / height
        sw = wide / width
        for i in range(high):
            for j in range(wide):
                x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
                y = int(j / sw + 0.5)
                emptyImage[i, j] = img[x, y]
    else:
        emptyImage = cv2.resize(img, (high, wide), interpolation=cv2.INTER_NEAREST)
    return emptyImage


plt.subplot(221)
gray_img = gray(img)
plt.imshow(gray_img, cmap='gray')

plt.subplot(222)
binary_img = black_white(img)
plt.imshow(binary_img, cmap='gray')

zoom = resize(img,800,800)
cv2.imshow("zoom_image", zoom)
cv2.imshow("original_image", img)
# cv2.waitKey(0)
plt.show()