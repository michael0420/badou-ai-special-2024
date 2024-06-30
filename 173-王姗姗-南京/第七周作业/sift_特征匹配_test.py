import cv2
import numpy as np


# 通过连线的方式比较两张图片的关键点
def drawMatchesKnn(img1, kp1, img2, kp2, goodMatches):
    # 获取图片的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 两个图像合并后的结果，
    result = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # 指定两个图像在结果中的位置，
    result[:h1, :w1] = img1
    result[:h2, w1:w1 + w2] = img2

    # 获取关键点
    p1 = [kpp.queryIdx for kpp in goodMatches]
    p2 = [kpp.trainIdx for kpp in goodMatches]

    # 将关键点坐标转换为整数类型并进行平移操作。
    post1 = np.int32([kp1[i].pt for i in p1])
    post2 = np.int32([kp2[i].pt for i in p2]) + (w1, 0)

    # 在图像上画出红色线段
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", result)


# 获取需要匹配的两张图片
img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")

# 创建sift特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 使用sift检测灰度图中的关键点和描述符，并返回关键点和描述符矩阵
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 基于暴力匹配的方式进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2)

# 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)

# 将符合标准的特征添加到goodMatch中
goodMatches = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatches.append(m)

drawMatchesKnn(img1, kp1, img2, kp2, goodMatches)

cv2.waitKey(0)
cv2.destroyAllWindows()
