import cv2 as cv
import numpy as np


def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):  # 绘制画线_detail
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)  # 创建一张空白图像，放下两个图像
    vis[:h1, :w1] = img1  # vis左半侧的值为img1，意思就是img1放在左边
    vis[:h2, w1:w1+w2] = img2  # img2放在vis右边，像素点坐标都是以左上角为原点！
    # print(vis[:h2, w1:w1+w2])  # 两个print展示了vis右边空白的值为img2
    # print(img2)

    p1 = [kpp.queryIdx for kpp in goodMatch]  # DMatch.queryIdx ：查询图像中描述符的索引。特征点在查询图像中的位置。
    p2 = [kpp.trainIdx for kpp in goodMatch]  # DMatch.trainIdx ： 目标图像中描述符的索引。特征点在目标图像中的位置。
    # p1和p2是索引值
    post1 = np.int32([kp1[pp].pt for pp in p1])  # .pt是KeyPoint类的一个属性，用于表示特征点的坐标。
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)  # 每一个坐标值都加上(w1, 0)，是为了在vis图像的右半侧
    # post1 = np.int32([kp1[m.queryIdx].pt for m in goodMatch])  # 这一行可以顶替15和18行
    # post2 = np.int32([kp2[m.trainIdx].pt for m in goodMatch])  # 这一行可以顶替16和19行

    for (x1, y1), (x2, y2) in zip(post1, post2):  # zip()函数：解包过程（下有详解）
        cv.line(vis, (x1, y1), (x2, y2), (0, 255, 0))  # 画线

    cv.namedWindow('match', cv.WINDOW_NORMAL)  # cv.WINDOW_NORMAL：用户可以调整窗口的大小
    cv.imshow('match', vis)


img1 = cv.imread('E:/Desktop/jianli/iphone1.png')
img2 = cv.imread('E:/Desktop/jianli/iphone2.png')

# 1、创建SIFT对象
sift = cv.xfeatures2d.SIFT_create()
# sift = cv.SURF()

# 2、使用SIFT对象的方法(detectAndCompute方法)检测关键点和计算描述子(关键点描述)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 每个特征点(关键点kp1或者kp2)本身也具有以下属性：.pt:关键点的点坐标，是像素坐标；.size:标出该点的直径大小(该关键点邻域直径大小)
# .angle：角度，表示关键点的方向，值为[0,360)，负值表示不使用；.response表示响应强度，选择响应最强的关键点

# 3、特征点匹配(为每个关键点返回k个最佳匹配)
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)  # bf.knnMatch()返回值matches是一个 DMatch 对象列表,下面有详解

# 4、对于描述符之间的距离的结果进行阈值的判断， 在阈值范围里的，匹配出来
goodMatch = []
for m, n in matches:
    # print('m:\n', m)
    # print('n:\n', n)
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)


drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch)

# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, goodMatch, None, flags=2)
# cv.imshow('match', img3)  # 如果 k 等于 2，就会为每个关键点绘制两条最佳匹配直线。

cv.waitKey(0)
cv.destroyAllWindows()
