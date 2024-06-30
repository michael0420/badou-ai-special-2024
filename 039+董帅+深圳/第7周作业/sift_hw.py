import cv2
import numpy as np

#read image
img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray_img,None)

img_with_keypoints = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints',img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
