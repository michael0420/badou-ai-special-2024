# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2

img = cv2.imread('photo1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Sort contours by area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Initialize document contour
docCnt = None

# Find document contour
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        docCnt = approx
        break

# Draw circles on document contour points
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

# Display image
cv2.imshow('img', img)
cv2.waitKey(0)
