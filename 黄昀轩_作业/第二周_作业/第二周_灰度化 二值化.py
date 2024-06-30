"""
@author: huangyunxuan

彩色图像的灰度化、二值化
"""
import cv2
import numpy as np
from skimage.color import rgb2gray

# 灰度化

img = cv2.imread("lenna.png")
w, h = img.shape[:2]
img_gray = np.zeros([w, h], img.dtype)
for i in range(w):
    for j in range(h):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

print(img_gray)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)

#灰度化 调用接口
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("img_gray",img_gray)
# cv2.waitKey(0)

#二值化
img = cv2.imread("lenna.png")
img_gray = rgb2gray(img)
print(img_gray)
w, h = img_gray.shape[:2]
print(w, h)
img_b = np.zeros([w, h], img_gray.dtype)
for i in range(w):
    for j in range(h):
        if (img_gray[i, j] < 0.5):  #条件是判断输入图像
            img_b[i, j] = 0
        else:
            img_b[i, j] = 1



print(img_b)
cv2.imshow("img_b", img_b)
cv2.waitKey(0)

