from matplotlib import pyplot as plt
import cv2
import numpy as np


#先灰度化
img = cv2.imread("lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#均衡化
img_Equal = cv2.equalizeHist(img_gray)

# 直方图
hist = cv2.calcHist([img_Equal],[0],None,[256],[0,256])

plt.figure()
plt.hist(img_Equal.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, img_Equal]))
cv2.waitKey(0)