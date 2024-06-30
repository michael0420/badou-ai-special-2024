import cv2
import numpy as np

def drawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    img = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)
    img[:h1, :w1] = img1
    img[:h2, w1:w1+w2] = img2

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    pt1 = np.int32([kp1[pp].pt for pp in p1])
    pt2 = np.int32([kp2[pp].pt for pp in p2]) + (w1,0)

    match_points = []
    for (x1, y1), (x2, y2) in zip(pt1, pt2):
        match_points.append(((x1, y1), (x2, y2)))
        cv2.line(img, (x1, y1), (x2, y2), color=(0,0,255))
    print(match_points)
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow('match', img)

img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#BFmatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k=2)  #将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。

#找出最佳匹配
goodMatch = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch[:20])
cv2.waitKey(0)
cv2.destroyAllWindows()
