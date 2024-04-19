import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png")
# 图像灰度化
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 图像均衡化
dst = cv2.equalizeHist(gray_img)

# 输出直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

# 使用matplotlib库来创建一个新的图形窗口
plt.figure()
# 使用matplotlib的hist函数来计算并显示图像dst的直方图
plt.hist(dst.ravel(), 256)
plt.show()
# hstack函数水平堆叠2个图像
cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst]))
cv2.waitKey(0)

# # 彩色图像直方图均衡化
# img = cv2.imread("lenna.png")
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge(bH, gH, rH)
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)