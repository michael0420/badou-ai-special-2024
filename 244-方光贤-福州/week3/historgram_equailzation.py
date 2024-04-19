import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
# images 图像指针 channels 通道[0] mask 无掩码 histSize 分成256份 ranges遍历范围
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
#revel()将图像转化为一维数组
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 彩色图像直方图均衡化
img1 = cv2.imread("lenna.png", 1)
cv2.imshow("src", img1)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
