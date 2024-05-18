import cv2
import matplotlib.pyplot as plt
import numpy as np
# 获取灰度图像
img = cv2.imread('lenna.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

# 展示
plt.figure(1)
plt.hist(gray.ravel(),256)

plt.figure(2)
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow('hitsgram equalizer',np.hstack([gray,dst]))
cv2.waitKey()

# 彩色图像直方图均衡化
img = cv2.imread('lenna.png',1)
b,g,r = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
dst = cv2.merge((bh,gh,rh))
cv2.imshow('img',img)
cv2.imshow('dst_rgb',dst)
cv2.waitKey()