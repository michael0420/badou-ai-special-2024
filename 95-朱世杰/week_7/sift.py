"""
实现SIFT
@Author: zsj
"""
import cv2
import numpy as np


def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 横向拼接图形
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    # 获取匹配点在两张图上的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


# 读取需要做对比的两个灰度图
img1_gray = cv2.imread("iphone1.png", 0)
img2_gray = cv2.imread("iphone2.png", 0)

# 创建SIFT特征检测器实例
sift = cv2.xfeatures2d.SIFT_create()

# 获取两张图的关键点及对应描述信息
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 创建BFMatcher实例
bf = cv2.BFMatcher(cv2.NORM_L2)
# 使用BFMatcher执行knnMatch，为每个查询点找到两个最近邻匹配
matches = bf.knnMatch(des1, des2, k=2)

# 筛选良好匹配
goodMatch = []
for m, n in matches:
    # 如果第一个邻居的距离小于第二个邻居的0.5倍，则认为是好的匹配
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# 绘制匹配点的连线
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch)

cv2.waitKey(0)
cv2.destroyAllWindows()
