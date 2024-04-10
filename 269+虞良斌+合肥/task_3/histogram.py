
import cv2
import numpy as np


'''直方图均衡化'''

img = cv2.imread("../lenna.png")

# 获取灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2 的均衡化函数
dst = cv2.equalizeHist(gray)

# 将两个或更多的数组沿着水平轴（即列）连接起来
cv2.imshow("Histogram", np.hstack([gray, dst]))
cv2.waitKey(0)