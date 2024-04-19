import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)

img = cv2.imread('lenna.png')
h, w, c = img.shape
img_nearest = np.zeros([1000, 1000, 3], img.dtype)
m = 1000 / h
n = 1000 / w
for i in range(1000):
    for j in range(1000):
        x = int(i / m + 0.5)
        y = int(j / n + 0.5)
        img_nearest[i, j] = img[x, y]

plt.subplot(222)
plt.imshow(img_nearest)


plt.show()

