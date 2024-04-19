# -*- coding: utf-8 -*-
"""
@author: jinlong

彩色图像的灰度化、二值化
"""
# 如何快速了解这些库都是干什么的，能干什么？
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化 1
img = cv2.imread("lenna.png")
h, w, c = img.shape[:3]  # 获取图片的high和wide（img的shape属性有三个参数：高，宽，通道）
print("picture:lenna","high=", h, "width=", w, "channels=", c, sep=" ", end=" 这是shape属性的三个参数\n")

# 如何获取帮助文档，例如zeros的说明
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        # m是RGB数据数组的指针，m[0]为R通道的值，以此类推
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
        #print(m)

print("image show gray: \n", img_gray)
cv2.imshow("image show gray", img_gray)





# subplot函数用于指定划分方式和位置进行绘图
plt.subplot(221) # 表示将整个图像窗口分为2行2列, 当前位置为1.
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)


# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222) # 表示将整个图像窗口分为2行2列, 当前位置为2.
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)


# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223) # 表示将整个图像窗口分为2行2列, 当前位置为3.
plt.imshow(img_binary, cmap='gray')
plt.show()
