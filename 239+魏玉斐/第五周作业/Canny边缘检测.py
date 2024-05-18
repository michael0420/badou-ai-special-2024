import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('lenna.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊来平滑图像
blurred = cv2.GaussianBlur(gray, (5, 5), 0.2)

# 调用Canny函数, 设置最小阈值和最大阈值为30和150
edges = cv2.Canny(blurred, 30, 150)

cv2.imshow('edges', edges)
cv2.waitKey(0)

