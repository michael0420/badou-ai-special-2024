import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 多通道转为单通道
img = cv2.imread("lenna.png")
cv2.imshow("rgb", img)
h, w = img.shape[:2]
img_gray = np.zeros((h, w), img.dtype)
print(img_gray.shape)
print(img_gray)
for i in range(h):
    for j in range(w):
        img_arr = img[i, j]
        img_gray[i, j] = int(0.11*img_arr[0] + 0.59*img_arr[1] + 0.3*img_arr[2])

cv2.imshow("rgb2gray", img_gray)
print(img_gray)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 二值化
for i in range(h):
    for j in range(w):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')


plt.show()

cv2.waitKey(0)