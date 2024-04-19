# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray


# 灰度化
#方法1
img = cv2.imread('lenna.png')
# h,w = img.shape[:2]
# img_gray = np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i,j]
#         # cv读入：BGR
#         img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#
# print(m)
# print(img_gray)
# print('image gray shows: %s'%img_gray)
# cv2.imshow('gray',img_gray)
# cv2.waitKey(0)

# 方法2(简洁）
img_gray = rgb2gray(img)
print(img_gray)
cv2.imshow('gray',img_gray)
cv2.waitKey(0)


# 二值化
# 方法1
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if(img_gray[i,j]<=0.5):
#             img_gray[i,j] = 0
#         else:
#             img_gray[i,j] = 1
# img_binary = img_gray


# 方法2（简洁）
img_binary = np.where(img_gray >= 0.5,1,0)

plt.subplot(221)  # 两行两列的第一个
img = plt.imread('lenna.png')
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show() # 显示图像