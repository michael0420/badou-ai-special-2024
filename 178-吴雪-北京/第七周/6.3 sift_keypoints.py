"""
SIFT (Scale-Invariant Feature Transform)是一种经典的特征提取方法
"""
import cv2 as cv

img = cv.imread('E:/Desktop/jianli/lenna.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 1、创建SIFT对象
sift = cv.xfeatures2d.SIFT_create()
# 2、使用SIFT对象的方法(detectAndCompute方法)检测关键点和计算描述子（关键点描述）
keypoints, descriptor = sift.detectAndCompute(gray, None)

# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
# (对每一个特征点绘制带大小和方向的关键点图形)
img = cv.drawKeypoints(image=img, keypoints=keypoints, outImage=img,
                       color=(51, 163, 236),
                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('sift_keypoints', img)
cv.waitKey(0)
cv.destroyAllWindows()