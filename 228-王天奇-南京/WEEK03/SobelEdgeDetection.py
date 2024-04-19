import numpy as np
import cv2

wtq_img = cv2.imread('lenna.png')

x = cv2.Sobel(wtq_img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(wtq_img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

result_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("SobelEdgeDetection", result_img)
cv2.waitKey(0)