# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2

image = cv2.imread("alex.jpg", 0)  # cv::ImreadModes 0 IMREAD_GRAYSCALE

imageX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
imageY = cv2.Sobel(image, cv2.CV_16S, 0, 1)

imageAbsX = cv2.convertScaleAbs(imageX)
imageAbsY = cv2.convertScaleAbs(imageY)

dst = cv2.addWeighted(imageAbsX, 0.5, imageAbsY, 0.5, 0)
dst2 = cv2.addWeighted(imageAbsX, 10, imageAbsY, 10, 0)

cv2.imshow("absX", imageAbsX)
cv2.imshow("absY", imageAbsY)

cv2.imshow("Result", dst)
cv2.imshow("Result2", dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()
