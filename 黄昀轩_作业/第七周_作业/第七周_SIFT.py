import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img1 = cv.imread("../iphone1.jpg")
img2 = cv.imread("../iphone2.jpg")

sift = cv.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good=[]
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img3)
plt.show()






















#
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# good=[]
#
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(img3)
# plt.show()



