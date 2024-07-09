"""
彩色图像的灰度化、二值化
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

# 二值化
img_binary = np.where(img_gray >= 128, 1, 0)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')

plt.show()
