#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', img)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 1, 0)

absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

cv2.imshow('absx', absx)
cv2.imshow('absy', absy)
cv2.imshow('result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

