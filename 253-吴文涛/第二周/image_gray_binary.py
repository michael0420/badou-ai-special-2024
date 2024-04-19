import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#灰度化
img = cv2.imread("image/lenna.png")
# print(type(img.shape)) 为元组类型 (512, 512, 3)
h,w = img.shape[:2] #获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)
# 函数作用：作用返回来一个给定形状和类型的用0填充的数组；
# 函数释义：zeros(shape, dtype=float, order=‘C’) shape:形状 dtype:数据类型，可选参数，默认numpy.float64
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
print("image show gray: %s" % img_gray)
#cv2.imshow("image show gray", img_gray)
#cv2.waitKey(0)#无限时间停留窗口

#plt 绘制图片 2行2列
plt.subplot(221)
img = plt.imread("image/lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
#plt.show()

#二值化
#   for j in range(cols):
#        if (img_gray[i, j] <= 0.5):
#            img_gray[i, j] = 0
#        else:
#            img_gray[i, j] = 1
img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()