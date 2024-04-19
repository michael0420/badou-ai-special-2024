import cv2
import numpy
import matplotlib.pyplot as plt


# 获取灰度图像
image = cv2.imread("lenna.png", 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray_image)  # 对灰度图像进行直方图均衡化处理，使图像的直方图分布更均匀

# 计算直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])  # 计算直方图，使用OpenCV的calcHist函数。这里是对均衡化后的图像dst计算直方图。

# 用Matplotlib绘制直方图
plt.figure()  # 创建一个新的Matplotlib图形
plt.hist(dst.ravel(), 256)  # 绘制直方图，dst.ravel()将二维数组dst转换为一维，256表示直方图的bin数量。
plt.show()

cv2.imshow("Histogram Equalization", numpy.hstack([gray_image, dst]))  # numpy.hstack()用于将原始灰度图像和均衡化后的图像水平拼接。
cv2.waitKey(0)

# 彩色图像直方图均衡化
# bgr_image = cv2.imread("lenna.png", 1)
# cv2.imshow("src", bgr_image)
#
# # 彩色图像均衡化  需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(bgr_image)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)