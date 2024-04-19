import cv2
import matplotlib.pyplot as plt
import numpy as np

# 加载图像
img = cv2.imread("lenna.png")

# 将原图转化为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均值化
equalized_img = cv2.equalizeHist(gray_img)

# 统计图像的直方图
hist = cv2.calcHist([equalized_img], [0], None, [256], [0,256])
# 创建画布
plt.figure()
# 画直方图 第一个值必须是一维数组 .ravel()：将数组维度拉成一维数组，256：直方图的柱数bin
plt.hist(equalized_img.ravel(), 256)
plt.show()

# 显示灰度图和直方图均值化后的图
# cv2.imshow("gray img",gray_img)
# cv2.imshow("equalized img",equalized_img)
cv2.imshow('Histogram Equalization', np.hstack([gray_img, equalized_img]))

cv2.waitKey(0)
cv2.destroyAllWindows()

