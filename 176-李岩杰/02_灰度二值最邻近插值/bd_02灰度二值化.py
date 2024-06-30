# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from PIL import Image
m = []
# 灰度化
img = cv2.imread("smallhill.jpg")
# h, w = img.shape[:2]  # 获取图片的high和wide
# img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]  # 取出当前high和wide中的BGR坐标
#         img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

# print(m)
# print(img_gray)
# print("image show gray: %s" % img_gray)
# cv2.imshow("image show gray", img_gray)

plt.subplot(221)
img = plt.imread("smallhill.jpg")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image hill----")
print(img)

# 灰度化
# img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image grayhill----")
print(img_gray)

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

img_binary = np.where(img_gray >= 128, 1, 0)
print("----- image_binary hill------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
