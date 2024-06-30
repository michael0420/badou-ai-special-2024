import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    """
    拼接两张图，并将特征点连线
    """
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 新的图片大小
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1 : w1 + w2] = img2_gray

    # 两站图的点
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 两张图的位置
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 注意坐标点的变化
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 连线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


import os
from __init__ import current_directory, cv_imread

img_path1 = os.path.join(current_directory, "img", "iphone1.png")
img_path2 = os.path.join(current_directory, "img", "iphone2.png")


def get_feature():

    img1 = cv_imread(img_path1)
    img2 = cv_imread(img_path2)

    print(img1.shape)
    print(img2.shape)

    sift = cv2.SIFT.create()
    # sift = cv2.SIFT_create()
    # sift = cv2.SURF()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # opencv中knnMatch是一种蛮力匹配
    # 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        # 取0.35的时候，可以把错误的斜线去除
        if m.distance < 0.35 * n.distance:
            goodMatch.append(m)

    # goodMatch = [m for m, n in matches]

    drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch[:30])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_feature()
