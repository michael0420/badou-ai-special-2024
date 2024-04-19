"""
直方图均衡化包括灰度图 和彩色图

"""

import  cv2
import numpy as np
import matplotlib.pyplot as plt

##灰度图直方图均衡化
imread = cv2.imread("../lenna.png",0)
# gray = cv2.cvtColor(imread, cv2.COLOR_BGR2GRAY)

dst=cv2.equalizeHist(imread)
cv2.imshow("gray hist",np.hstack([imread,dst]))
# # 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()



##彩色图直方图均衡化
imread = cv2.imread("../lenna.png")
#彩色图的直方图均衡化需要解析每一个通道
b,g,r = cv2.split(imread)
bH=cv2.equalizeHist(b)
gH=cv2.equalizeHist(g)
rH=cv2.equalizeHist(r)
c_dst=cv2.merge([bH,gH,rH])
cv2.imshow("color dist",np.hstack([imread,c_dst]))

cv2.waitKey(0)
