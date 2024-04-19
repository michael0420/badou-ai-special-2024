import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# 手动灰度图像直方图均值化
def Histogram_Equalization_gray(img):
    # 图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_h, gray_w = gray.shape[:2]

    # 提取并统计灰度信息
    hist_src = np.zeros(256)
    for gray_y in range(gray_h):
        for gray_x in range(gray_w):
            hist_src[gray[gray_y, gray_x]] += 1

    # 计算均衡化后灰度值
    hist_dst = np.zeros(256)
    sum = 0
    for i in range(256):
        sum += (hist_src[i]*256)/(gray_h*gray_w)
        hist_dst[i] =int(sum-1)

     # 映射
    img_dst = gray.copy()
    for dst_y in range(gray_h):
        for dst_x in range(gray_w):
            img_dst[dst_y,dst_x] = hist_dst[gray[dst_y,dst_x]]

    return img_dst


img = cv2.imread('Grace.jpg',1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 手动灰度图像直方均值化
img_dst = Histogram_Equalization_gray(img)
#自带灰度图像直方均值化
img_dst2 = cv2.equalizeHist(img_gray)
hist = cv2.calcHist([img_dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(img_dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray,img_dst,img_dst2]))
cv2.waitKey(0)

# 彩色图像直方图均衡化

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", np.hstack([img,result]))

cv2.waitKey(0)