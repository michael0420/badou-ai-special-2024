import cv2
import numpy as np

# 定义一个函数，用于绘制knn匹配的结果
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    # 展示两张图片和匹配线
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray
    # 提取匹配点在原图像上的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    # 获取匹配点在原图像上的位置
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    # 绘制匹配线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

# 创建一个SIFT特征检测器对象
sift = cv2.xfeatures2d.SIFT_create()
# 使用SIFT检测关键点和计算描述符
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 创建一个BFMatcher对象，使用L2范数进行匹配
bf = cv2.BFMatcher(cv2.NORM_L2)

# 使用knnMatch进行匹配，k设置为2，即每个特征点匹配两个最佳结果
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个
matches = bf.knnMatch(des1, des2, k=2)

# 筛选好的匹配点（使用Lowe's ratio test）
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# 绘制前20个匹配结果并显示
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
