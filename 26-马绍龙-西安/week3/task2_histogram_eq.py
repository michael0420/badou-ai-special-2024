import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

img = cv2.imread('../week2/lenna.png', 1)  # flag为1，表示读取图片为BGR图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)  # 直接调用接口进行图片均衡化处理

hist_pre = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_post = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.subplot(211)
plt.hist(gray.ravel(), bins=256)  # ravel() 是 Numpy将多维数组展平成一维数组 的方法。[[1, 2, 3],[4, 5, 6]]变为[1, 2, 3, 4, 5, 6]

plt.subplot(212)
plt.hist(dst.ravel(), 256, [0, 256])
plt.show()

cv2.imshow('compare', np.hstack((gray, dst)))
cv2.waitKey(0)



# 彩色图像均衡化,将三个通道分别进行均衡化
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

b_post = cv2.equalizeHist(b)
g_post = cv2.equalizeHist(g)
r_post = cv2.equalizeHist(r)

dst = cv2.merge((b_post, g_post, r_post))

cv2.imshow('color', numpy.hstack([img, dst]))
cv2.waitKey(0)
