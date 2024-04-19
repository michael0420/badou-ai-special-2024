import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)
cv2.imshow("img", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", gray)


# 灰度图像 直方图均衡化
dst = cv2.equalizeHist(gray)
cv2.imshow("dst", dst)
# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))//将两图合并成一张图显示

# 彩色图像直方图均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

#直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure() #创建新的图像窗口
plt.hist(dst.ravel(), 256)
plt.xlim([0,256])
plt.show()

cv2.waitKey(0)
