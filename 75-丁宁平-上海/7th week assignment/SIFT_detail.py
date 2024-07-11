# coding=utf-8

import cv2
import numpy as np

# print(cv2.__version__)    # 检查版本

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, temp):
    row1, col1 = img1_gray.shape[:2]
    row2, col2 = img2_gray.shape[:2]

    vis = np.zeros((max(row1, row2), col1 + col2, 3), dtype=np.uint8)
    vis[:row1, :col1] = img1_gray
    vis[:row2, col1:col1 + col2] = img2_gray

    p1 = [kpp.queryIdx for kpp in temp]
    p2 = [kpp.trainIdx for kpp in temp]

    pos1 = np.int32([kp1[pp].pt for pp in p1])
    pos2 = np.int32([kp2[pp].pt for pp in p2]) + (col1, 0)

    for (x1, y1), (x2, y2) in zip(pos1, pos2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

temp = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        temp.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, temp[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
