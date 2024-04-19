import cv2
import numpy as np
import matplotlib.pyplot as plt


# ====灰度图像直方图均衡化====
img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imageGray', gray)

dst = cv2.equalizeHist(gray)

plt.figure('a')
plt.hist(gray.ravel(), 256)

plt.figure('b')
plt.hist(dst.ravel(), 256)

hist = cv2.calcHist([dst], [0], None, [256], [0, 255])
plt.figure('c')
plt.plot(hist)
plt.show()

cv2.imshow('histogramEqualization', np.hstack([gray, dst]))
cv2.waitKey(0)

# ====彩色图像直方图均衡化====
img = cv2.imread('lenna.png', 1)
cv2.imshow('src', img)

b, g, r = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

plt.figure('rgb_b')
plt.hist(bh.ravel(), 256)

plt.figure('rgb_g')
plt.hist(gh.ravel(), 256)

plt.figure('rgb_r')
plt.hist(rh.ravel(), 256)
plt.show()

result = cv2.merge((bh, gh, rh))
cv2.imshow('dst_rgb', result)
cv2.waitKey(0)
