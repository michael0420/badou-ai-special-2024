import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("lenna.png")
# 灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
img_hist_equal = cv2.equalizeHist(img_gray)

# 计算直方图
hist = cv2.calcHist([img_hist_equal], [0], None, [256], [0, 255])


plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
plt.axis("off")

plt.subplot(222)
plt.hist(img_gray.ravel(), 256)

plt.subplot(223)
plt.imshow(img_hist_equal, cmap='gray')
plt.axis('off')

plt.subplot(224)
plt.hist(img_hist_equal.ravel(), 256)

plt.show()


plt.plot(hist)
plt.show()


# 彩色图像直方图均衡化
# 彩色图像图层分离
b, g, r = cv2.split(img)

# 分别进行直方图均衡化
b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)

# 合并通道
img_hist = cv2.merge((b_hist, g_hist, r_hist))

cv2.imshow("contrast", np.hstack([img, img_hist]))
cv2.waitKey()

# 彩色图片直方图
b_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
g_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
r_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

plt.plot(b_hist, color='b')
plt.plot(g_hist, color='g')
plt.plot(r_hist, color='r')
plt.show()