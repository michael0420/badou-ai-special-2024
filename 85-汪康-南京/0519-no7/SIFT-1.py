# -*- coding: utf-8 -*-
'''@Time: 2024/5/19 16:20

'''
import cv2
import numpy as np
img = cv2.imread("../lenna.png")
gray = cv2.imread("../lenna.png",0)

sift = cv2.xfeatures2d.SIFT_create()
keypoints,descriptor = sift.detectAndCompute(gray,None)
img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
                        color=(51,163,236))
cv2.imshow('sift_keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()