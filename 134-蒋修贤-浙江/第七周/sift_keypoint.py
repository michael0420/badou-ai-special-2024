#!/usr/bin/env python3
import cv2
import numpy as np

img = cv2.imread('lenna.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT.create()
keypoint,descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoint, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0,0,0))

cv2.imshow('sift_keypoint',img)
cv2.waitKey()
cv2.destroyAllWindows()