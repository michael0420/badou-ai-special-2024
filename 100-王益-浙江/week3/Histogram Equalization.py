import cv2
import numpy as np
import matplotlib.pyplot as plt


# 手动实现直方图均衡化
def histogram_equalization(img):
    hist = np.zeros(256)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            hist[img[i, j]] += 1

    hist = hist / (height * width)
    sum = 0
    for i in range(256):
        sum += hist[i]
        hist[i] = sum * 256 - 1

    img_eq = img.copy()
    for i in range(height):
        for j in range(width):
            img_eq[i, j] = hist[img[i, j]]

    return img_eq


img = cv2.imread('img/lenna.png', 1)
plt.figure()
plt.hist(img.ravel(), 256)
plt.show()
blue, green, red = cv2.split(img)
img_eq = cv2.merge((histogram_equalization(blue), histogram_equalization(green), histogram_equalization(red)))
img_eq_2 = cv2.merge((cv2.equalizeHist(blue), cv2.equalizeHist(green), cv2.equalizeHist(red)))
plt.figure()
plt.hist(img_eq.ravel(), 256)
plt.show()
cv2.imshow('Original Image', img)
# 手动实现的直方图均衡化
cv2.imshow('Histogram Equalized Image', img_eq)
# 对比OpenCV的直方图均衡化
cv2.imshow('OpenCV Histogram Equalized Image', img_eq_2)
cv2.waitKey(0)
