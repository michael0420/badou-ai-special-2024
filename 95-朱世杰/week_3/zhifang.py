"""
直方图均衡化
@Author: zsj
"""
import cv2
from matplotlib import pyplot as plt


img = cv2.imread("../lenna.png", 1)
# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
dst = cv2.equalizeHist(gray)

# 绘制图像
plt.figure()
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(gray, cmap='gray')
plt.subplot(223)
plt.imshow(dst, cmap='gray')

# 绘制直方图
plt.subplot(224)
plt.hist(dst.ravel(), 256)

# 彩色图像均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))

# 绘制直方图
plt.figure()
plt.subplot(221)
plt.hist(bH.ravel(), 256)
plt.subplot(222)
plt.hist(gH.ravel(), 256)
plt.subplot(223)
plt.hist(rH.ravel(), 256)

# 绘制图像
plt.subplot(224)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result_rgb)
plt.show()