import cv2
import numpy as np


img = cv2.imread("lenna.png")

# 横竖两方向进行特征边缘提取
img_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
img_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 将像素值变为0-255范围
img_x = cv2.convertScaleAbs(img_x)
img_y = cv2.convertScaleAbs(img_y)

# cv2.imshow("x", img_x)
# cv2.imshow("y", img_y)

# 将两个方向综合
img_sobel = cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)
cv2.imshow("integer", np.hstack([img_x, img_y, img_sobel]))
cv2.waitKey()