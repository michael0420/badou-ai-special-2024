
import numpy as np
import cv2
import matplotlib.pyplot as plt

#  直方图均衡化

#加载图片
img = cv2.imread("lenna.png")
#灰度
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)

#直方图均衡化
dst = cv2.equalizeHist(gray)

#创建一张新图
plt.figure()
#绘制直方图函数
# dst.ravel() 多维数组转换为1维数组
# plt.hist 绘制直方图函数
plt.hist(dst.ravel(), 256)
plt.show()

# 展示灰度和均衡化的图片
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)