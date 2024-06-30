import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    # 获取第一张和第二张图像的高度和宽度
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 初始化一个空白的可视化图像，大小足够放下并排的两张图片
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # 将两张图片粘贴到可视化图像上
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    # 提取匹配中查询图像和训练图像的关键点索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 转换关键点索引为实际坐标位置
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 对第二幅图的关键点位置加上第一幅图的宽度偏移
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 在可视化图像上画线连接匹配的关键点
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))  # 红色线连接匹配点

    # 创建并显示匹配结果的窗口
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)
    cv2.waitKey(0)  # 等待按键后关闭窗口


# 读取灰度图像
img1_gray = cv2.imread("iphone1.png", cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread("iphone2.png", cv2.IMREAD_GRAYSCALE)

# 使用SIFT创建特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 检测并计算两幅图像的特征点及其描述符
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 创建BFMatcher对象，使用L2范数作为距离度量
bf = cv2.BFMatcher(cv2.NORM_L2)

# 使用knnMatch进行匹配，返回每一对最佳和次佳匹配
#opencv中knnMatch是一种蛮力匹配
#将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)

# 存储良好匹配（最佳匹配与次佳匹配的距离小于一定阈值的）
goodMatch = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # 这里的阈值被调整为0.75
        goodMatch.append(m)

# 调用函数显示匹配结果
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch)

cv2.waitKey(0)
cv2.destroyAllWindows()
