#!/usr/bin/env python3
'''
1.灰度化
2.均衡化
3.直方图绘制
'''
import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread("lenna.jpeg")
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)

cv2.imshow('h e',np.hstack([gray,dst]))

plt.figure()
plt.subplot(221)
plt.hist(gray.ravel(),256)
plt.subplot(222)
plt.hist(dst.ravel(),256)
plt.show()
cv2.waitKey()

img = cv2.imread('lenna.jpeg')
(b,g,r) = cv2.split(img)
r = cv2.merge((cv2.equalizeHist(b),cv2.equalizeHist(g),cv2.equalizeHist(r)))
cv2.imshow('rgb',np.hstack([img,r]))

plt.figure()
plt.subplot(221)
plt.hist(img.ravel(),256)
plt.subplot(222)
plt.hist(r.ravel(),256)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()