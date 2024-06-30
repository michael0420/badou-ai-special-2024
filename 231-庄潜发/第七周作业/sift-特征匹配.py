"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/26 21:52
"""
import cv2
import numpy as np


def main():
    # 读取两张图片
    img1 = cv2.imread("iphone1.png")
    img2 = cv2.imread("iphone2.png")

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    # 分别检测两张图像的关键点和计算描述符
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

    # 创建BFMatcher对象, 特征匹配器, 它可以计算两张图像描述符之间的距离
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # 使用KNN进行特征匹配, 计算描述符之间的距离, k=2表示找到两个最近的匹配
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # 筛选出好的匹配,通过比较第一个匹配和第二个匹配的距离,
    # 如果第一个匹配的距离小于第二个匹配的距离的50%,
    # 那么我们认为这是一个"好的匹配"
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)
    # 绘制匹配结果, goodMatch[:20]表示取前面20个匹配项, 一般图片会有很多匹配项, 便宜分析
    drawMatchesKnn_cv2(img1, keypoint1, img2, keypoint2, goodMatch[:20])


def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    """
    创建一个新图像，将两个图像并排放置，然后根据匹配的关键点坐标绘制线连接匹配点。
    """
    # 计算两张图片的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 创建一个空白图像，用于绘制两张图片及其匹配线
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    # 提取匹配的关键点索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 获取关键点的位置
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 绘制匹配线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    # 显示结果
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
