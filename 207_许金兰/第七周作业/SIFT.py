"""
@author: 207-xujinlan
sift关键点检测
"""

import cv2

# 1.读入图片，并进行灰度化
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2.SIFT实例化
sift = cv2.xfeatures2d.SIFT_create()

# 3.检测关键点并计算特征描述子
keypoints, descriptor = sift.detectAndCompute(img_gray, None)

# 4.画出关键点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

# 5.图片展示
cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
