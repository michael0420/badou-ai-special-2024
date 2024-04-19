import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取图片
img = cv2.imread('../0327/lenna.png')
# 灰度化
# gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('huiduhua',gray_img)
# cv2.waitKey()
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray_img)
# cv2.imshow('junhenghua',dst)
# cv2.waitKey()

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
print(hist)
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# 对比灰度图和均衡化后的图
cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst]))
cv2.waitKey(0)
