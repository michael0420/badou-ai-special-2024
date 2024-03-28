from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread('Grace.jpg')
h,w= img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
# print(type(img.dtype))
for i in range(h):
    for j in range(w):
        p = img[i,j]
        img_gray[i,j] = p[0]*0.11+p[1]*0.59+p[2]*0.3

# print(img_gray)
# cv2.imshow('photo',img_gray)
# cv2.waitKey(0)

# 二值化
h1,w1 = img_gray.shape
img_binary = np.zeros([h1,w1])
for i in range(h1):
    for j in range(w1):
        p2 = (img_gray[i,j])/255
        if p2 <= 0.3:
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 255

# cv2.imshow('image binary',img_binary)
# # cv2.waitKey(0)

# 图片显示
plt.subplot(221)
img2 = plt.imread('Grace.jpg')
plt.imshow(img2)

plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()