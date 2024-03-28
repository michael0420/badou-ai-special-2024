"""
@Date    : 2024/3/27 20:08
@Author  : zwx
@Version : 1.0
@Desc    : RGB图片进行灰度化、二值化Demo
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

print('====================  灰度化:基于原生代码 =======================')
img = cv2.imread('../data/input/lenna.png')  # 使用 opencv 库读图片， 获取到的是 BGR 图像
h, w, z = img.shape[:3]  # 获取图片的 高（512）、宽（512）、通道数（3）
img_gray = np.zeros([h, w], img.dtype)  # 创建相同大小、通道数的画布
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出 BGR 坐标的像素点
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 按照浮点计算公式 Gray= R0.3 + G0.59 + B0.11 计算灰度图像值

print("gray1_value: %s" % img_gray)
cv2.imshow("gray_img1", img_gray)
cv2.waitKey(0)  # 等待函数，关闭图片即结束程序

print('===================== 灰度化:基于调用接口 ======================')
plt.subplot(221)
img = plt.imread("../data/input/lenna.png")  # 使用 matplotlib 库读图片，读取到的是 RGB 图像
plt.imshow(img)  # 原图

plt.subplot(222)
rgb_gray = rgb2gray(img)  # API灰度化
print("gray2_value: %s" % rgb_gray)
plt.imshow(rgb_gray, 'gray')  # 展示灰度化
# plt.waitforbuttonpress(0)

print('===================== 二值化:基于原生代码 ======================')
plt.subplot(223)
img = plt.imread("../data/input/lenna.png")
rgb_gray = rgb2gray(img)
h, w = rgb_gray.shape  # 在灰度图基础上进行二值化，灰度图为单通道，无通道数
for i in range(h):
    for j in range(w):
        if rgb_gray[i, j] <= 0.5:
            rgb_gray[i, j] = 0  # 黑色
        else:
            rgb_gray[i, j] = 1  # 白色

print("binary_value1: %s" % rgb_gray)
plt.imshow(rgb_gray, 'gray')
# plt.waitforbuttonpress(0)

print('===================== 二值化:基于调用接口 ======================')
plt.subplot(224)
img = plt.imread("../data/input/lenna.png")
rgb_gray = rgb2gray(img)
binary_img = np.where(rgb_gray <= 0.5, 0, 1)  # API二值化
print("binary_value2: %s" % binary_img)
plt.imshow(binary_img, 'gray')
plt.waitforbuttonpress(0)  # 等待函数，关闭图片即结束程序
