"""
@author: 207-xujinlan
边缘提取，cv2实现canny
"""

import cv2

pic_path = 'lenna.png'
# 1.读入图片
img = cv2.imread(pic_path)
# 2.图片灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 3.canny边缘检测
img_canny = cv2.Canny(img_gray, 150, 250)
cv2.imshow('canny', img_canny)
cv2.waitKey(0)
