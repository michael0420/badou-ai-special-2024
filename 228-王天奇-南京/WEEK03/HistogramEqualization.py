import numpy as np
import cv2

wtq_img = cv2.imread('lenna.png')
wtq_img_g = cv2.cvtColor(wtq_img, cv2.COLOR_BGR2GRAY)
# 灰度图直方图均衡化
wtq_img_g_e = cv2.equalizeHist(wtq_img_g)
# 生成直方图
hist = cv2.calcHist([wtq_img_g_e], [0], None, [256], [0,256])
cv2.imshow("Histogram Equalization", np.hstack([wtq_img_g, wtq_img_g_e]))


# 彩色直方图均衡化
(b, g, r) = cv2.split(wtq_img)
be = cv2.equalizeHist(b)
ge = cv2.equalizeHist(g)
re = cv2.equalizeHist(r)
wtq_img_e = cv2.merge((be,ge,re))
cv2.imshow("Histogram Equalization", np.hstack([wtq_img, wtq_img_e]))
cv2.waitKey(0)