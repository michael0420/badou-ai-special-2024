#!/usr/bin/env python
# encoding=gbk

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
from matplotlib import pyplot as plt  # 从Matplotlib库中导入pyplot模块，用于绘图

img = cv2.imread("lenna.png", 1)  # 读取名为"lenna.png"的彩色图像
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 将彩色图像转换为灰度图像

'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子核的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''

# Sobel导数运算   这两个图像分别表示了原始图像中水平和垂直方向上的边缘信息  ksize这里设置为3，表示使用3x3的Sobel算子核
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 对x方向进行Sobel导数运算
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # 对y方向进行Sobel导数运算

# Laplace 算子
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  # 使用Laplace算子检测图像边缘

# Canny 算子
img_canny = cv2.Canny(img_gray, 100, 150)  # 使用Canny算子进行边缘检测

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")  # 绘制原始灰度图像
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")  # 绘制Sobel x方向导数图像
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")  # 绘制Sobel y方向导数图像
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")  # 绘制Laplace算子检测的边缘图像
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")  # 绘制Canny边缘检测结果图像
plt.show()  # 显示绘制的图像
