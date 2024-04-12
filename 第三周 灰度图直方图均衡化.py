import cv2
import numpy as np
from matplotlib import pyplot as plt


rawImg = cv2.imread("/Users/mac/Desktop/tuanzi.jpg")

grayImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)  # 转成灰度图

targetImg = cv2.equalizeHist(grayImg)  # 对灰度图进行直方图均衡化

'''
绘制直方图
cv2.calcHist(images,channels,mask,histSize,ranges)
图像，通道，遮掩图像，展示多少个条柱，像素值范围
'''
#hist = cv2.calcHist([targetImg],[0],None,[256],[0,256])

plt.figure()
plt.hist(targetImg.ravel(), 256)
#plt.hist(hist, 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([grayImg, targetImg]))
cv2.waitKey(0)