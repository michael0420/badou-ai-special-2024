import cv2
import numpy as np

img = cv2.imread('iphone1.png')
gray = cv2.imread('iphone1.png',0)

sift = cv2.SIFT.create()
kp,ds = sift.detectAndCompute(gray,None)
img_kp = cv2.drawKeypoints(image=img,keypoints=kp,outImage=img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(123,255,154))

cv2.imshow("keyPoints",img_kp)
cv2.waitKey()