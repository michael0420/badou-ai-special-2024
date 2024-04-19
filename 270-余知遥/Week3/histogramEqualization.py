import cv2
import numpy as np



img = cv2.imread('./lenna.png')
(b, g, r) = cv2.split(img)#彩图
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow('original', img)
cv2.imshow('equalized', result)
cv2.waitKey(0)