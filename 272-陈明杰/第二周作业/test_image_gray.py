# -*- coding: utf-8 -*-
"""
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# # 灰度化
# def imgGray(img: np.ndarray):
#     high, width = img.shape[:2]  # 获取img矩阵的高度和宽度
#     # 创建一个和img一样的的单通道矩阵
#     img_gray = np.zeros((high, width), img.dtype)
#     for i in range(high):
#         for j in range(width):
#             # 取出每个像素的RGB值，但是这里要特别注意，opencv
#             # 对于读进来的图片的通道排列是BGR而不是主流的RGB
#             m = img[i, j]
#             # 求BGR三通道的加权平均和，0.11*B+0.59*G+0.3*R
#             img_gray[i, j] = int(0.11 * m[0] + 0.59 * m[1] + 0.3 * m[2])
#     return img_gray
#
#
# def main():
#     # 调用cv2模块读取图片，cv2模块相当于一个类，里面有很多成员方法
#     img = cv2.imread("lenna.png")
#     img_gray = imgGray(img)
#     cv2.imshow("original", img)  # 显示原图
#     cv2.imshow("gray", img_gray)  # 显示灰度化之后的图
#     print(img)
#     print(img_gray)
#     print("image show gray: %s" % img_gray)
#     cv2.waitKey(0)  # 让终端停止下来，方便观察
#
#
# main()

# #二值化
# def imgBinary(img):
#     rows,cols=img.shape[0:2]
#     for i in range(rows):
#         for j in range(cols):
#             if img[i,j]<=0.5:
#                 img[i,j]=0.0
#             else:
#                 img[i,j]=1.0
#     return img
#
#
# # 二值化
# img_binary=imgBinary(img_gray)


# plt.subplot(221)创建了一个 2x2 的子图网格，
# 并将当前绘图区域设置为该网格的左上角子图
plt.subplot(221)
img = plt.imread("lenna.png")
# 把原图放到子图网格对应位置中
plt.imshow(img)

# 用库中的方法实现灰度化
img_gray = rgb2gray(img)
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(222)
# 把灰度化的图放到子图网格对应位置中
plt.imshow(img_gray, cmap='gray')

# 用库中的方法实现二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap="gray")
# 显示整个2*2子网个
plt.show()

