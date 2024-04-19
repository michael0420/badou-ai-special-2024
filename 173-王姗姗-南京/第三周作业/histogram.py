import cv2
import numpy as np
from matplotlib import pyplot as plt

#  直方图均衡化

#  获取灰度图像,单通道图像
img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# 灰度图像直方图均衡化
out = cv2.equalizeHist(gray)

# 统计图像的直方图
hist = cv2.calcHist([out], [0], None, [256], [0, 256])

plt.figure()
plt.hist(out.ravel(), 256)
#  展示直方图
plt.show()

# 原图与均衡化之后的图像对比展示
cv2.imshow("histogram ", np.hstack([gray, out]))
cv2.waitKey(0)

'''
#  彩色图像直方图均衡化
img = cv2.imread('lenna.png',1)
cv2.imshow("src",img)
# 彩色图像均衡化，需要分解通道，对每一个通道均衡化
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并每一个通道
result = cv2.merge((bH,gH,rH))
cv2.imshow("out_img",result)
cv2.waitKey(0)
'''
