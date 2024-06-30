import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

# 特征匹配
img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

sift = cv2.SIFT.create()  # 特征检测器对象
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)  # 返回 检测到的 SIFT 特征点的坐标和特征描述子
kp2, des2 = sift.detectAndCompute(img2_gray, None)  # 返回 检测到的 SIFT 特征点的坐标和特征描述子


# BFmatcher with default parms  创建一个暴力匹配器对象
'''
BFMatcher是OpenCV中的一个特征匹配算法，全称为Brute-Force Matcher，它使用暴力搜索的方式，对于每一个图像中的特征点，都会在另一幅图像中进行搜索，
找到最匹配的特征点。
normType：表示特征点匹配的距离度量标准，有以下几种类型可供选择：
cv2.NORM_L1：L1 范数，即绝对值的和。
cv2.NORM_L2：L2 范数，即平方和的平方根。
cv2.NORM_HAMMING：汉明距离，即两个特征点描述子之间的汉明距离。
'''
bf = cv2.BFMatcher(cv2.NORM_L2)

# opencv中knnMatch是一种蛮力匹配
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
'''
`cv2.BFMatcher`的`knnMatch()`方法来进行特征点的 K 近邻匹配。
`bf.knnMatch(des1, des2, k=2)`的参数说明如下：
- `des1`：表示第一张图像的特征描述子，是一个二维数组。
- `des2`：表示第二张图像的特征描述子，也是一个二维数组。
- `k=2`：表示匹配的近邻数量，即返回前 K 个最接近的匹配点。 第一张图的一个点对应两一张图的2个点
`knnMatch()`方法会返回一个列表`matches`，其中包含了所有匹配到的特征点对。每个特征点对由两个元组组成，分别表示第一张图像和第二张图像中的特征点坐标。
                                     如果没有找到匹配的特征点对，则返回空列表。
'''
matches = bf.knnMatch(des1, des2, k=2)


'''
从匹配结果`matches`中筛选出较好的匹配点，并存储在`goodMatch`列表中。代码中使用了`cv2.BFMatcher`的`distance`属性来计算特征点之间的距离。
遍历了`matches`列表中的每一对匹配点， m 和 n 分别是第一张图像和第二张图像中的特征点。然后计算 m 和 n 的距离，用 m.distance 和 n.distance 分别表示。
接着，利用条件判断语句`if m.distance < 0.50 * n.distance`筛选出较好的匹配点。这里的 0.50 是一个经验值，可以根据实际情况进行调整。
     如果 m 的距离小于 0.50 倍的 n 的距离，就认为 m 是一个较好的匹配点，将其添加到`goodMatch`列表中。
`goodMatch`列表中存储了经过筛选的较好的匹配点。
'''
goodMatch = []
for m, n in matches:
    # 第一张图的一个点对应两一张图的2个点，若图1中点到图2中点1的距离 < 0.5倍图1中点到图2中点2的距离，则 图1中点 和 图2中点1 近似度比较高
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

'''
使用 OpenCV 库中的`drawMatchesKnn()`函数来绘制 K 近邻匹配结果。它将图像`img1_gray`和`img2_gray`中的特征点`kp1`和`kp2`进行匹配，
           并根据`goodMatch`列表中的前 20 个匹配结果在图像上绘制出来。
具体来说，`drawMatchesKnn()`函数的参数有以下几个：
- `img1`：第一张图像，这里是`img1_gray`。
- `kp1`：第一张图像中的特征点列表，这里是`kp1`。
- `img2`：第二张图像，这里是`img2_gray`。
- `kp2`：第二张图像中的特征点列表，这里是`kp2`。
- `matches`：匹配结果列表，这里是`goodMatch[:20]`，表示取`goodMatch`列表的前 20 个元素。

希望这些解释对你有帮助！
'''
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
