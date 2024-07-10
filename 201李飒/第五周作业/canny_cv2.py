import cv2
import numpy as np

# 直接调用cv2的canny 函数
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 100, 200))
cv2.waitKey()