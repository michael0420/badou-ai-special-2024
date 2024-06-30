import cv2
import numpy as np

# 定义一个函数用于绘制匹配结果
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    # 获取两个图像的高度和宽度
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 创建一个空白图像用于显示两个图像并排
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    # 从goodMatch中提取匹配的关键点索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 获取匹配点的坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 在vis图像中绘制匹配的线条
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0))

    # 显示匹配结果
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

# 读取两个待匹配的图像
img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

# 创建SIFT特征检测器
sift = cv2.SIFT_create()

# 计算两幅图像的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 创建暴力匹配器，使用L2范数
bf = cv2.BFMatcher(cv2.NORM_L2)

# 使用knn算法匹配描述符，找出每个匹配中的前两个最相似的点
matches = bf.knnMatch(des1, des2, k=2)

# 根据距离选择好的匹配
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# 绘制前20个好的匹配结果
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

# 等待用户响应并销毁所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
