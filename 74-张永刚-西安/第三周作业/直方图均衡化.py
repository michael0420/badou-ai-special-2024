"""
@author: 74+张永刚
直方图均衡化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("../第二周作业/image/img/lenna.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("",gray)
# cv2.waitKey(0)

# 灰度直方图均衡化
dst = cv2.equalizeHist(gray)
# cv2.imshow("",dst)
# cv2.waitKey(0)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)

# 彩色直方图均衡化
img2 = cv2.imread("../第二周作业/image/img/lenna.png",1)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b,g,r) = cv2.split(img2)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bh,gh,rh))
cv2.imshow("Histogram",result)
cv2.waitKey(0)

