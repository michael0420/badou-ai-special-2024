"""
@author 193杨婷
灰度图像、彩色图像直方图均衡化
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)  # 参数为1，表示以RGB模式加载图像，忽略Alpha通道
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
print(dst)  # 每个元素代表图像中对应像素的灰度值
"""
生成直方图函数
images：表示输入的图像，数据类型是Python的列表。对于单通道图像，应将其作为一个元素放在列表中，对于多通道图像，每个通道应作为一个元素放在列表中。
channels：表示要统计直方图的通道索引，如果是灰度图像，该值为0；如果是RGB图像，可以指定要计算哪个通道的直方图。
mask：表示掩模图像，如果不使用掩模，则为None。
histSize：表示直方图的大小（bin的数量）。
ranges：表示像素值的范围，通常为[0,256]。
"""
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
print(hist)  # 显示了直方图中每个灰度级别的像素点数量

plt.figure()
plt.hist(dst.ravel(), 256)  # dst.ravel()将图像dst转换为一个一维数组,256(bin)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))  # np.hstack()函数是NumPy中用于将多个数组在水平方向叠加在一起的函数
cv2.waitKey(0)


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
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)