import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("lenna.png")
#灰度图均衡化
# grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# #直方图均衡化
# dst = cv.equalizeHist(grayImg)
#
# hist = cv.calcHist([dst], [0], None, [256], [0, 256])
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv.imshow("Test", np.hstack([grayImg, dst]))
# cv.waitKey()
#RGB图均衡化
(r, g, b) = cv.split(img)
rh = cv.equalizeHist(r)
gh = cv.equalizeHist(g)
bh = cv.equalizeHist(b)
newImg = cv.merge((rh, gh, bh))
cv.imshow("Test", np.hstack([img, newImg]))
cv.waitKey()
