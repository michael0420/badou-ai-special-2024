import cv2
import numpy as np

img = cv2.imread("iphone1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptor = sift.detectAndCompute(gray, None)#得到keypoints, descriptor:438*128矩阵，应该是完整描述

# cv2.drawKeypoints(source, output, keypoints(通过sift.detectAndCompute()得到), flag, color))
# 将图像的所有关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))


cv2.imshow('SIFT_out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()