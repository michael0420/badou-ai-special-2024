"""
彩色图像的灰度化，二值化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from PIL import Image

# 1）灰度化-用浮点算法
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# print (m)
# print (img_gray)
# print("image show gray: %s"%img_gray)

cv2.imshow("image show gray",img_gray)
# 延迟窗口显示时间
# cv2.waitKey()

# 显示原图片
img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
# plt.show()


# 2）灰度化-用rgb2gray
img_gray = rgb2gray(img)
# 转换颜色空间
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
# plt.show()

# 3) 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if(img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
# plt.subplot(223)
# plt.imshow(img_gray, cmap='gray')
# plt.show()

img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()