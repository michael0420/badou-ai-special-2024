"""
@author: QiPiaoYang

彩色图像灰度化、二值化

"""

import cv2
import numpy as np


img = cv2.imread("../lenna.png")
cv2.imshow("srcImg", img)
cv2.waitKey(0)

# 灰度化

h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("grayImg", img_gray)
cv2.waitKey(0)

# 二值化
rows, cols = img_gray.shape[:2]
img_binary = np.zeros([rows, cols], img.dtype)
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] < 128:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255

# ret, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("binaryImg", img_binary)
cv2.waitKey(0)
