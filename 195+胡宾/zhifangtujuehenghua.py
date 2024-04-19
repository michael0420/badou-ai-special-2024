import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
# 单通道均衡化

img = cv2.imread('lenna.png')
# 灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像均衡化
target = cv2.equalizeHist(img_gray)
# 直方图
hist = cv2.calcHist([target],[0],None,[256],[0,256])

plt.figure()
plt.hist(target.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, target]))
cv2.waitKey(0)





