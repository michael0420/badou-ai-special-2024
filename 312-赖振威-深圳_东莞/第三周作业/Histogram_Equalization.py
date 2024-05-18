#!/usr/bin/env python
# encoding=gbk

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
from matplotlib import pyplot as plt  # 从matplotlib库中导入pyplot模块

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取彩色图像
img = cv2.imread("lenna.png", 1)  # 读取彩色图像  将flag从1改为0也可以变为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
cv2.imshow("image_gray", gray)  # 显示名为image_gray灰度图像
cv2.waitKey(0)  # 等待按键事件

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)  # 对灰度图像进行直方图均衡化

# 计算直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()  # 创建新的图形
plt.hist(dst.ravel(), 256)  # 绘制直方图
plt.show()  # 显示直方图
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))  # 显示原始图像和均衡化后的图像
cv2.waitKey(0)  # 等待按键事件

# ----------------------------------------------------------------------------------------------------------------------
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)  # 读取彩色图像
cv2.imshow("src", img)  # 显示原始彩图像

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)  # 分离彩色图像的通道
bH = cv2.equalizeHist(b)  # 对蓝色通道进行直方图均衡化
gH = cv2.equalizeHist(g)  # 对绿色通道进行直方图均衡化
rH = cv2.equalizeHist(r)  # 对红色通道进行直方图均衡化
# 合并每一个通道
result = cv2.merge((bH, gH, rH))  # 合并处理后的通道
cv2.imshow("dst_rgb", result)  # 显示均衡化后的彩色图像

cv2.waitKey(0)  # 等待按键事件
