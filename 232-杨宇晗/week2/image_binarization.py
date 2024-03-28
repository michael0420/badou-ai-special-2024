"""
@author: Hanley-Yang

彩色图像的二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化处理
img = cv2.imread("shangri-la.jpg")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
# 遍历灰度化每个像素值
# for i in range(h):
#     for j in range(w):
#         init = img[i,j]
#         img_gray[i,j] = int(init[0]*0.11 + init[1]*0.59 + init[2]*0.3)
img_gray = rgb2gray(img)

#二值化
rows, cols = img_gray.shape
img_binary = np.zeros([h,w],img.dtype)
for i in range(rows):
    for j in range(cols):
        img_binary[i,j] = int(img_gray[i,j]+0.5)
print("---image gray----")
print(img_gray)

#img_binary = np.where(img_gray >= 0.5, 1, 0)

print("---image binary---")
print(img_binary)
print(img_binary.shape)

#显示原始图片在画布上方
plt.subplot(211)
img = plt.imread("shangri-la.jpg")
plt.imshow(img)

#显示二值化图片在画布下方
plt.subplot(212)
plt.imshow(img_binary, cmap='gray')
plt.show()

