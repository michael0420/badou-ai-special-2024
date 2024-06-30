"""
SIFT 特征点检测与描述
"""

import cv2

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp,
						flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
						color=(51, 163, 236))
cv2.imshow('sift keypoints', img)
cv2.waitKey(0)
cv2.destroyWindow()
