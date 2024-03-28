'''
    第二周作业：临近插值法实现图像上采
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')


def upsampling(img):    # python函数要写在方法调用前
    h, w, c = img.shape
    targetImage = np.zeros([1000, 1000, c], np.uint8)
    a = 1000 / h        # 获得尺寸比值
    b = 1000 / w
    for i in range(1000):       # 遍历虚拟像素点
        for j in range(1000):
            x = int(i / a + 0.5)     # 获得原图像素点
            y = int(j / b + 0.5)
            targetImage[i, j] = img[x, y]   # 虚拟像素点临近赋值
    return targetImage


pic = upsampling(img)
cv2.imshow('origin_pic', img)
cv2.imshow('nearest_neighbor_interpolation', pic)
cv2.waitKey(0)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(pic)
plt.show()

print(pic)
