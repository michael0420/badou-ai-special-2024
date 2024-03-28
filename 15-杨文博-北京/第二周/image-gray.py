import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray

#图像灰度化原理
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)

for i in range(h):
    for j in range(w):
        BGR = img[i,j]
        img_gray[i,j] = int(BGR[0]*0.11 + BGR[1]*0.59 + BGR [2]*0.3)   #opencv大坑顺序BGR

#直接调用接口
img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
plt.imshow(img_gray1, cmap='gray')

#二值化
H, W = img_gray.shape
for i in range(H):
    for j in range(W):
        if (img_gray[i, j] /255 <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)

plt.subplot(224)
plt.imshow(img_binary, cmap='gray')
plt.show()