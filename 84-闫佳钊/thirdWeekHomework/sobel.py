import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('lenna.png', 0)
x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# cv2.imshow('x', x)
# cv2.imshow('y', y)
cv2.imshow('absX', absX)
cv2.imshow('absY', absY)
cv2.imshow('result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
