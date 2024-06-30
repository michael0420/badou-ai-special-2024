import cv2
import numpy as np

if __name__ == '__main__':

    img1_gray = cv2.imread("iphone1.png")
    img2_gray = cv2.imread("iphone2.png")

    # sift = cv2.SIFT()
    sift = cv2.SIFT_create()
    # sift = cv2.SURF()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)  # 得到第一个图的 关键点和描述子
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # opencv中knnMatch是一种蛮力匹配
    # 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append([m])

    # vis = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, goodMatch, None, (0, 0, 255)) # 一对一
    vis = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, goodMatch, None, (0, 0, 255))  # 一对多

    cv2.imshow("match", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
