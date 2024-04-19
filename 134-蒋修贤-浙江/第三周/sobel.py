#!/usr/bin/env python3

import numpy as np
import cv2

img = cv2.imread("lenna.jpeg")

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x = cv2.Sobel(grey, cv2.CV_16S, 1, 0)
y = cv2.Sobel(grey, cv2.CV_16S, 0, 1)

aX = cv2.convertScaleAbs(x)
aY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(aX, 0.5, aY, 0.5, 0)

cv2.imshow('aX', aX)
cv2.imshow('aY', aY)

cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()