import cv2 as cv
import numpy as np

img = cv.imread('lenna.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keypoints, desc = sift.detectAndCompute(gray, None)
# print(keypoints)
# print(desc)
#绘制关键点的圆圈和方向
img = cv.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('newImg', img)
cv.waitKey()
cv.destroyAllWindows()