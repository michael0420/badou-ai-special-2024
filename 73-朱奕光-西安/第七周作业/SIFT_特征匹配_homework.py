import cv2
import numpy as np

phone1 = cv2.imread('photo1.png')
phone2 = cv2.imread('photo2.png')

"""
定义sift
"""
sift = cv2.SIFT_create()

"""
计算两个图像的关键点列表和描述符矩阵
"""
kp1, des1 = sift.detectAndCompute(phone1, None)
kp2, des2 = sift.detectAndCompute(phone2, None)

"""
暴力匹配相似度最高关键点
"""
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

goodMatch = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

"""
匹配点可视化
"""
h1, w1 = phone1.shape[:2]
h2, w2 = phone2.shape[:2]
dst = np.zeros([max(h1, h2), w1 + w2, 3], np.uint8)
dst[:h1, :w1] = phone1
dst[:h2, w1:w1 + w2] = phone2

p1 = [kp.queryIdx for kp in goodMatch]
p2 = [kp.trainIdx for kp in goodMatch]

post1 = np.int32([kp1[n].pt for n in p1])
post2 = np.int32([kp2[n].pt for n in p2])+(w1,0)

for (x1, y1), (x2, y2) in zip(post1,post2):
    cv2.line(dst, (x1, y1), (x2, y2), (199,237,204))
cv2.imshow('dst', dst)
cv2.waitKey(0)