# encoding=UTF-8

import cv2
from matplotlib import pyplot as plt
import numpy as np

# 获取灰度图片
img = cv2.imread("lenna.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # 变换为单通道
cv2.imshow("gray",gray)
cv2.waitKey()
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)    # GBK 报错SyntaxError: encoding problem: gbk

# 直方图,cv2.calcHist()函数统计图像直方图信息
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()                # 创建一个图像
plt.hist(dst.ravel(),256)   # plt.hist()绘制直方图,src.ravel()可以将二维图像拉平为一维数组
# plt.show()                # 闪退。以下是解决方案
plt.ion()       # 显示直方图
plt.pause(15)   # 显示秒数
plt.close()
# 对比均衡化前后图片
# np.vstack():在竖直方向上堆叠     np.hstack():在水平方向上平铺
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)

# 彩色图直方图均衡化
img = cv2.imread("lenna.png",1)     # 获取图片
cv2.imshow("src",img)
cv2.waitKey()
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b,g,r) = cv2.split(img)        # 对图像进行拆分
# 分别均衡化 B G R
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bh,gh,rh))
cv2.imshow("均衡化合并后的图",result)
cv2.waitKey()