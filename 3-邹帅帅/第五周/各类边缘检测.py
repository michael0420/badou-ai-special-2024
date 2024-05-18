
#!/usr/bin/env python
# encoding=UTF-8

import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png', 1)
# img = cv2.imread('lenna.png', 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Soble 算子
# Soble函数求导会有负值以及大于255的值 
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

#转回uint8形式
abs_sobel_x = cv2.convertScaleAbs(img_sobel_x)
abs_sobel_y = cv2.convertScaleAbs(img_sobel_y)

#加权组合
img_sobel = cv2.addWeighted(abs_sobel_x, 0,5, abs_sobel_y, 0.5, 0)

# Laplace 算子  
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# Canny 算子 
img_canny = cv2.Canny(img_gray, 100, 150)

# subplot 创建2x3 的窗口 使用 imshow 显示图像，设置颜色映射为灰度 ,title 设置名称， axis 闭坐标轴
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("img_sobel_x")
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("img_sobel_y")
plt.subplot(234), plt.imshow(img_sobel, "gray"), plt.title("img_sobel")
plt.subplot(235), plt.imshow(img_laplace, "gray"), plt.title("img_laplace")
plt.subplot(236), plt.imshow(img_canny, "gray"), plt.title("img_canny")