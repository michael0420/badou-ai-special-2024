import cv2
import numpy as np
def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1,w1 = img1_gray.shape[:2]
    h2,w2 = img2_gray.shape[:2]
    #合并2张图大小
    result = np.zeros((max(h1,h2),w1 + w2, 3), np.uint8)
    result[:h1, :w1] = img1_gray
    result[:h2,w1:w1+w2] = img2_gray

    #特征点查询图像索引位置
    p1 = [t.queryIdx for t in goodMatch]
    #特征点训练图像索引位置
    p2 = [t.trainIdx for t in goodMatch]

    #从kp1中获取特征点的坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 从kp1中获取特征点的坐标，并且偏移w1个坐标
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", result)


img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")


#sift
sift = cv2.xfeatures2d.SIFT_create()
k1, d1 = sift.detectAndCompute(img1_gray, None)
k2, d2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)
#opencv中knnMatch是一种蛮力匹配
#将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
matches = bf.knnMatch(d1, d2, k = 2)

goodMatch = []

for m,n in matches:
    if m.distance < 0.5* n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, k1, img2_gray, k2, goodMatch)
cv2.waitKey(0)
cv2.destroyAllWindows()





