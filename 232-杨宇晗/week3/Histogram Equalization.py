"""
@author: Hanley-Yang

equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
"""

import cv2
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread('Euphonium.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像的直方图均衡化
dst = cv2.equalizeHist(grayImg)

# 获取直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

#输出原灰度图和均衡化后的灰度图的直方图，进行对比
plt.figure()
plt.subplot(221)
plt.hist(grayImg.ravel(),256)
plt.hist(dst.ravel(),256)

plt.subplot(222)
plt.imshow(dst,cmap="gray")

# cv2.imshow("Histogram Equalization",np.hstack([grayImg,dst]))
# cv2.waitKey(0)

# # 彩色图像直方图均衡化
# img = cv2.imread("Euphonium.png",1)
# cv2.imshow("src",img)

# 彩色图像均衡化，需要分解通道，对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

# 将图像从BGR转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.imshow(img_rgb)

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.subplot(224)
plt.imshow(result_rgb)

plt.show()



