import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png',1)  #1表示三通道彩图，0表示单通道灰度图

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(img_gray)   #equalizeHist—直方图均衡化  函数原型： equalizeHist(src, dst=None)
print(dst)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])  #灰度图就是0，如果是彩色图像可以是012分别对应BGR
print(hist)

cv2.imshow("Histogram Equalization", np.hstack([img_gray, dst])) #np.hstack将两个数组连接同一行合并在一起，np.vstack将两个数组连接同一行合并在一起
plt.figure()
plt.hist(dst.ravel(), 256)  #ravel()将array数组转换为一维数组
plt.show()


# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
# 直方图
hist = cv2.calcHist([result],[1],None,[256],[0,256])  #彩色图像可以是012分别对应BGR
cv2.imshow("dst_bgr",np.hstack([img, result]))
plt.figure()
plt.hist(result.ravel(), 256)  #ravel()将array数组转换为一维数组
plt.show()



