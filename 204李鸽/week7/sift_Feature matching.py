import cv2
import numpy as np

# 用于在两幅图像中的关键点之间绘制匹配线
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):        # 接受两个灰度图像img1_gray和img2_gray，关键点kp1和kp2，以及一组良好匹配goodMatch作为输入参数
    h1, w1 = img1_gray.shape[:2]                              # [:2] 表示对图像的高度和宽度切片操作，将返回图像的高度和宽度两个值
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)       # 创建一个高度为最大高度的、宽度为两幅图像宽度之和的空图像vis，通道数为3（RGB），使用NumPy库
    vis[:h1, :w1] = img1_gray                                 # 将第一幅图像img1_gray放置在vis图像的左上角
    vis[:h2, w1:w1 + w2] = img2_gray                          # w1:w1 + w2 表示从 w1 到 w1+w2 的宽度范围，将第二幅图像img2_gray放置在vis图像的右上角

    p1 = [kpp.queryIdx for kpp in goodMatch]                  # 从良好匹配goodMatch列表中提取查询索引
    p2 = [kpp.trainIdx for kpp in goodMatch]                  # 从良好匹配goodMatch列表中提取训练索引

    post1 = np.int32([kp1[pp].pt for pp in p1])               # 计算第一幅图像中与关键点对应的整数坐标点，kp2[pp].pt 表示获取关键点 kp2 中索引为 pp 的关键点，然后获取其坐标值（通常是 (x, y) 形式的点坐标）
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)     # 算第二幅图像中与关键点对应的整数坐标点，并调整x坐标加上w1，因为放在了右边

    # 接下来的代码块绘制了来自两幅图像中匹配关键点之间的连线

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)               # 使用OpenCV创建一个命名窗口用于显示结果
    cv2.imshow("match", vis)                                  # 在创建的窗口中显示带有匹配的图像

img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray, None)            # 使用 SIFT 特征检测器来检测图像 img1_gray 中的关键点，并且计算这些关键点的描述子。其中 kp1 是检测到的关键点列表，des1 是对应的描述子矩阵
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
# opencv中knnMatch是一种蛮力匹配，使用 BFMatcher 时，我们需要指定用来衡量距离的方法，例如欧式距离（L2 范数）
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，计算描述子之间的距离，找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)        # 使用 BFMatcher 对象 bf 来对 des1 和 des2 中的描述子进行匹配，k=2：取最近邻匹配前两个描述子

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:        # 第一个距离小于第二个最近距离的一半，说明这俩距离差距大，第一个距离贼小
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])      # 匹配结果中的前 20 个好的匹配点

cv2.waitKey(0)
cv2.destroyAllWindows()