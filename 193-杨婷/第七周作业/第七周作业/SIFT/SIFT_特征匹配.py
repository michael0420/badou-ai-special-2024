import cv2
import numpy as np

# 在两张图像上绘制关键点之间的匹配线
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]  # 获取第一张图像的高度和宽度
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)  # 创建一个新的空白图像vis，该图像足够大，可以容纳两张输入图像并排显示。
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]  # 从goodMatch中获取第一张图像的关键点索引。
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])  # 从第一张图像的关键点列表kp1中获取匹配关键点的坐标。
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))  # 绘制匹配线

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread('iphone1.png')
img2_gray = cv2.imread('iphone2.png')

sift = cv2.SIFT_create()  # 声明
kp1, des1 = sift.detectAndCompute(img1_gray, None)  # 搜索：对象kp1有什么属性？
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)  # 创建BFMatcher（蛮力匹配）对象,这里用欧氏距离
'''
一旦你创建了 BFMatcher 对象，你可以使用它的 match 或 knnMatch 方法来匹配特征。
match 方法：返回最佳匹配。
knnMatch 方法：返回每个描述符的 k 个最佳匹配。
'''
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，每个点找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50*n.distance:  # 说明前二的差距很大，第一的距离非常相似
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
