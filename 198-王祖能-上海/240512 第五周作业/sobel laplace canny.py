'''
sobel laplace canny几种接口结果对比
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0)
# 1. Sobel算子
img_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # CV_64F表示64位浮点型，CV_16S表示16位有符号整型, 接口原因不支持32S以上的整型
img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
print('Sobel_X算子后的矩阵为：\n', img_x)
print('Sobel_Y算子后的矩阵为：\n', img_y)
img_x, img_y = cv2.convertScaleAbs(img_x, cv2.COLOR_BGR2RGB), cv2.convertScaleAbs(img_y, cv2.COLOR_BGR2RGB)
'''
# convert这一步是否需要？？？
cv2.imread读的图是彩色图，要用BGR2RGB后，plt.imshow才能显示。是灰度图的话，直接用plt.imshow就能显示，不需要convert
是因为img_x跟img_y都是单通道数组，convert没起啥作用而已，有没有没区别
'''
print(img_x)
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(img_x, cmap='gray'), plt.title('Sobel_X')
plt.subplot(2, 3, 3), plt.imshow(img_y, cmap='gray'), plt.title('Sobel_Y')
'''
# dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
# ddepth是图像的深度，-1表示采用的是与原图像相同的深度, 目标图像的深度必须大于等于原图像的深度；
# dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
# dst是目标图像；
# ksize是Sobel算子的大小，必须为1、3、5、7。
# scale是缩放导数的比例常数，默认情况下没有伸缩系数；
# delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
# borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''

# 2. Laplace算子，空间锐化滤波操作，对噪声敏感，需要结合高斯滤波
img_Laplace = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
img_Laplace = cv2.convertScaleAbs(img_Laplace, cv2.COLOR_BGR2RGB)
print('Laplace算子后的矩阵为：\n', img_Laplace)
plt.subplot(2, 3, 4), plt.imshow(img_Laplace, cmap='gray'), plt.title('Laplacian')

# 3.Canny算子
img_Canny = cv2.Canny(img, 120, 280, apertureSize=3, L2gradient=True)
img_Canny = cv2.convertScaleAbs(img_Canny, cv2.COLOR_BGR2RGB)
print('Canny算子后的矩阵为：\n', img_Canny)
plt.subplot(2, 3, 5), plt.imshow(img_Canny, cmap='gray'), plt.title('Canny')

plt.show()
