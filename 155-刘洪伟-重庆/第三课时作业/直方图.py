# _*_ coding: UTF-8 _*_
# @Time: 2024/4/3 19:51
# @Author: iris
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('../data/lenna.png', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist)
plt.xlim([0, 256])

image = cv2.imread('../data/lenna.png')
channels = cv2.split(image)
colors = ('r', 'g', 'b')
for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()
