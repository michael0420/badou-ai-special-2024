import numpy as np
import cv2 as cv

img = cv.imread('lenna.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(img, kp, img,
                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                       color=(51, 163, 236))
cv.imshow('sift_keypoints', img)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.imwrite('sift_keypoints.jpg', img)