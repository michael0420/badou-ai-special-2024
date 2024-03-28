# _*_ coding: UTF-8 _*_
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = plt.imread('../data/lenna.png')
plt.subplot(2, 2, 1)
plt.imshow(image)

# 先做灰度化再进行二值化转换
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
for i in range(rows):
    for j in range(cols):
        if image_gray[i, j] <= 0.5:
            image_gray[i, j] = 0
        else:
            image_gray[i, j] = 1

plt.subplot(2, 2, 2)
plt.imshow(image_gray, cmap='gray')

img = np.where(image_gray <= 0.5, 0, 1)
plt.subplot(2, 2, 3)
plt.imshow(img, cmap='gray')

plt.show()
