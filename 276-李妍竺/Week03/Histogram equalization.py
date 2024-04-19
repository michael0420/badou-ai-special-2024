

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png', 1)  #(1: color; 0:gray; -1:unchanged)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# hist = cv2.calcHist([gray],[0],None,[256],[0,256])
# 1.灰度图像的直方图 2.使用的通道是第 0 个（灰度通道）(若为彩色：0，1，2对应B,G,R)
# 3.None表示指定掩码图像，即不显示像素为xx的点，None表示没有要掩盖的。
# 4.BIN的数目为 256. 5.像素值的范围为 0 到 255(开区间）。

# #dst.ravel() 将数组展平为一维数组，所有元素按行优先顺序排列成一个新的一维数组
# plt.hist(img.ravel(),256)， 256表示直方图的柱数

# 直方图均衡化
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# hist = cv2.calcHist([gray],[0],None,[256],[0,256])  #原灰度化的图

# 创建新图像
plt.figure()
plt.hist(dst.ravel(), 256)
# plt.hist(gray.ravel(), 256)  #原灰度化的图
plt.show()

# 对比图像
# np.hstack() 函数水平平铺两张图像 np.vstack() 竖直方向
cv2.imshow("Histogram Equavlization", np.vstack([gray, dst]))
cv2.waitKey(0)
# cv2.destroyAllWindows()



# ______________________________________________
# 彩色图像均衡化,需要分解通道,对每一个通道均衡化.
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.imshow('dst_rgb_and_original',np.hstack([result,img]))
cv2.waitKey(0)


