import cv2
import numpy as np
import matplotlib.pyplot as plt

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
imgGray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        imgGray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print("img show gray: %s" % imgGray)
plt.subplot(221)
imgPlt = plt.imread("lenna.png")
plt.imshow(imgPlt)
plt.subplot(222)
plt.imshow(imgGray, cmap='gray')

# 二值化
imgBinary = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        if (imgGray[i, j] / 255 <= 0.5):
            imgBinary[i, j] = 0
        else:
            imgBinary[i, j] = 1
print("img show binary: %s" % imgBinary)
plt.subplot(223)
plt.imshow(imgBinary, cmap='gray')
plt.show()
