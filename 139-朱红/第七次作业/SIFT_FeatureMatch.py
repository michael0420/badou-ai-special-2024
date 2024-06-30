import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('iphone1.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('iphone2.png', cv.IMREAD_GRAYSCALE)  # trainImage

# 初始化ORB detector
orb = cv.ORB_create()

# 用ORB找到关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建BFMatcher对象
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # 打开交叉验证

# 获取两个图像中的最佳匹配
matches = bf.match(des1, des2)

# 按照距离的升序对其进行排列
matches = sorted(matches, key = lambda x:x.distance)

# 画出前10个最佳匹配
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()