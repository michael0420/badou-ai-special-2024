import cv2
import numpy as np

import matplotlib.pyplot as plt


img = cv2.imread('lenna.png', 1) # 默认是1

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 注意是BGR

dst = cv2.equalizeHist(gray_img) # 对灰度图像进行直方图均衡化

# 使用cv2.calcHist()函数计算均衡化后的灰度图像的直方图
hist = cv2.calcHist([dst], [0], None, [256], [0,256]) # 读取dst图形，指定通道索引(灰度图像是0)，掩码图像None，直方图长度256，取值像素0-256

# plt.figure() # 创建一个新的图形窗口
# plt.hist(dst.ravel(), 256) # dst.ravel() 函数将二维数组 dst 展平为一维数组，然后 plt.hist() 函数接收这个一维数组作为输入，绘制直方图
# plt.show()
#
# cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst]))
# cv2.waitKey(0)


# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)



# 合并每一个通道
result = cv2.merge((bH, gH, rH))


plt.figure() # 创建一个新的图形窗口
plt.hist(histB.ravel(), 256) # dst.ravel() 函数将二维数组 dst 展平为一维数组，然后 plt.hist() 函数接收这个一维数组作为输入，绘制直方图
plt.show()
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
