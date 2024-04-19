# 图像直方图均衡化
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图像灰度化处理
# cv2.imshow("img_gray", img_gray)

# 灰度图像直方图均衡化
img_equ = cv2.equalizeHist(img_gray)
# cv2.imshow('img_equ', img_equ)

# 直方图
img_hist = cv2.calcHist([img_equ], [0], None, [256], [0, 256])

plt.figure()
plt.hist(img_equ.ravel(), 256)  # ravel()数组多维度拉成一维数组
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, img_equ]))  # np.hstack将参数元组的元素数组按水平方向进行叠加
cv2.waitKey(0)
