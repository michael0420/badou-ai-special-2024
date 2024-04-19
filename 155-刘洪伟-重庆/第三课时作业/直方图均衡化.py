# _*_ coding: UTF-8 _*_
# @Time: 2024/4/3 19:51
# @Author: iris
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('../data/lenna.png', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.subplot(2, 1, 1)
plt.hist(dst.ravel(), 256)

cv2.imshow("Histogram hist", np.hstack([gray, dst]))

(b, g, r) = cv2.split(image)
b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)
result = cv2.merge((b_hist, g_hist, r_hist))

channels = cv2.split(result)
colors = ('r', 'g', 'b')

for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.subplot(2, 1, 2)
    plt.plot(hist, color=color)

plt.show()

cv2.imshow('Histogram Equalization', np.hstack([image, result]))
cv2.waitKey(0)
