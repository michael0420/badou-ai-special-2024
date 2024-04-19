import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("../lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)    # 对gary进行直方图均衡化
cv2.imshow("Histogram Equalization", np.hstack([gray,dst]))     # np.hstack() 函数用于将两幅图像水平叠加显示
cv2.waitKey()

plt.figure()    # 创建新的图形窗口的函数
plt.hist(dst.ravel(),256)    # 绘制直方图, dst.ravel() 将 dst 图像的像素值展平为一维数组，256 表示直方图的区间数
plt.show()


# 彩色
img = cv2.imread("../roses-1566792_640.jpg")
(b,g,r) = cv2.split(img)
b_h = cv2.equalizeHist(b)
g_h = cv2.equalizeHist(g)
r_h = cv2.equalizeHist(r)

dst = cv2.merge((b_h,g_h,r_h))
cv2.imshow("dst",np.hstack([img,dst]))
cv2.waitKey()
