"""
@author: Hanley-Yang

彩色图像的灰度化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

#灰度化
img = cv2.imread("shangri-la.jpg")
h,w,c = img.shape[:3]   #获取图片的high,wide和channel
img_gray = np.zeros([h,w],img.dtype) #创建一张和当前图片大小一样的单通道图片
#二维循环，取出当前high和wide中的RGB坐标
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

#输出原始图片像素参数
print("---image shangri-la---")
#输出灰度化图片像素参数
print(m)
print("---image gray----")
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

#显示原始图片在画布上方
plt.subplot(211)
img = plt.imread("shangri-la.jpg")
plt.imshow(img)

#显示灰度化图片在画布下方
img_gray = rgb2gray(img)
plt.subplot(212)
plt.imshow(img_gray, cmap='gray')


plt.show()
