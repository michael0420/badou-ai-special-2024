import cv2
import numpy as np


def drawMatchesKnn_cv2(iphone1, kp1, iphone2, kp2, goodMatch):
    h1, w1 = iphone1.shape[:2]
    h2, w2 = iphone2.shape[:2]

    # 创建一个能同时放置两张图片的窗口
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = iphone1
    vis[:h2, w1:w1 + w2] = iphone2
    # cv2.imshow("vis", vis)
    # cv2.waitKey(0)

    # queryIdx 和 trainIdx 是 DMatch 对象中的两个重要属性，用于标识在两个不
    # 同的描述符集合（通常称为“查询集”和“训练集”）中特定描述符的索引。在特征匹
    # 配的场景中，我们通常有一个“查询”图像（或图像集）和一个“训练”图像（或图像集）。
    #
    # queryIdx：这个属性标识了查询描述符在查询描述符集合（例如 des1）中的索引。
    # 假设我们有一个包含多个描述符的列表 des1，queryIdx 会告诉我们当前 DMatch
    # 对象所代表的匹配关系中的查询描述符在 des1 中的位置。
    #
    # trainIdx：这个属性标识了训练描述符在训练描述符集合（例如 des2）中的索引。
    # 同样地，假设我们有一个包含多个描述符的列表 des2，trainIdx 会告诉我们当前
    # DMatch 对象所代表的匹配关系中的训练描述符在 des2 中的位置。
    #
    # 在特征匹配的过程中，我们通常会对查询描述符集合中的每一个描述符，在训练描述
    # 符集合中寻找最相似的描述符（或几个最相似的描述符，如使用 knnMatch 方法时）。
    # 一旦找到了匹配，我们就可以使用 queryIdx 和 trainIdx 来确定这些匹配描述符
    # 在各自集合中的位置，进而可以在原始图像中找到这些描述符所对应的位置（比如关键点）。
    #
    # 这种索引的标识方式使得我们可以轻松地在原始图像之间建立匹配关系，并进行后续的
    # 分析和处理，比如计算变换矩阵、拼接图像、目标跟踪等。
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 遍历找出20个匹配点的坐标，通过kp1[pp].pt可以找到kp1[pp]点的坐标
    # 在特征匹配和计算机视觉的上下文中，kp1 和 kp2 通常表示两组关键点（KeyPoints）的列表，
    # 这些关键点是通过某种特征检测算法（如SIFT、SURF、ORB等）在图像中检测到的。每个关键点
    # 对象（如OpenCV中的cv2.KeyPoint）通常包含有关该关键点的信息，如其在图像中的位置（坐标）、
    # 方向、大小等。
    # kp1[pp].pt 和 kp2[pp].pt 访问的是关键点对象的 .pt 属性，这个属性是一个元组，表示关
    # 键点在图像中的坐标（通常是(x, y)形式）。具体来说，x 是关键点的水平坐标（即列），y 是
    # 关键点的垂直坐标（即行）。
    # 这行代码正在创建一个NumPy数组（np.int32类型），其中包含了与p1列表（通常是一个关键点索
    # 引的列表）对应的关键点在kp1列表中的坐标。p1中的每个索引pp都被用来从kp1中获取对应的关键点，
    # 并取其坐标。
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 创建一个NumPy数组，但之后它还通过+ (w1, 0)将每个坐标向右平移了w1个像素（假设w1是一个整数，
    # 表示宽度）。这是因为如果kp2来自另一张图像，并且你想将这些关键点映射到与kp1相同的坐标系
    # 中（例如在图像拼接或注册的上下文中），可能需要将它们进行平移以匹配两张图像之间的相对位置。
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 在所有的(x1,y1)和(x2,y2)对应点之间连线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), color=(0, 0, 255))

    cv2.imshow("vis", vis)
    cv2.waitKey(0)


# 1、读取图片
iphone1 = cv2.imread("iphone1.png")
iphone2 = cv2.imread("iphone2.png")

# 2、初始化SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 3、检测特特征点和计算描述符
kp1, des1 = sift.detectAndCompute(iphone1, None)
kp2, dst2 = sift.detectAndCompute(iphone2, None)

# 在图像中绘制出特征点
iphone1_with_keyPoints = cv2.drawKeypoints(iphone1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
iphone2_with_keyPoints = cv2.drawKeypoints(iphone2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
iphone1_with_keyPoints = cv2.cvtColor(iphone1_with_keyPoints, cv2.COLOR_BGR2RGB)
iphone2_with_keyPoints = cv2.cvtColor(iphone2_with_keyPoints, cv2.COLOR_BGR2RGB)

# cv2.imshow("iphone1",iphone1_with_keyPoints)
# cv2.waitKey(0)
# cv2.imshow("iphone2",iphone2_with_keyPoints)
# cv2.waitKey(0)

# 创建了一个Brute-Force Matcher对象，并指定了使用L2范数（即欧几里得距离）来进
# 行描述符之间的比较。cv2.NORM_L2表示使用L2范数（欧几里得距离）来计算两个描述符
# 之间的相似度。
bf = cv2.BFMatcher(cv2.NORM_L2)
# Brute-Force Matcher对象（bf）来找到两组描述符（des1和des2）之间的最佳匹配。
# 这里使用了knnMatch方法，它返回每个描述符在另一组描述符中的k个最近邻。在这个例
# 子中，k被设置为2，所以它会为每个描述符在另一组描述符中找到两个最近的匹配。
# matches是一个列表，其中每个元素都是一个包含两个DMatch对象的列表。每个DMatch对
# 象代表一个匹配，包含查询描述符的索引、训练描述符的索引、它们之间的距离等信息。在
# 这个例子中，因为k=2，所以matches中的每个元素都包含两个DMatch对象，分别代表最近
# 和次近的匹配。
# 这种k-最近邻（kNN）匹配方法常用于特征匹配中的比率测试（ratio test）。在比率测试
# 中，如果最近邻与次近邻之间的距离之比小于某个阈值（如0.75），则认为这个匹配是好的。
# 这种方法可以帮助过滤掉那些由于噪声、重复结构或背景造成的误匹配。
matchs = bf.knnMatch(des1, dst2, k=2)

# matches 是一个列表，其中每个元素是一个包含两个 DMatch 对象的列表。
# 这两个 DMatch 对象分别代表从 des1 中的描述符到 des2 中的最近邻和
# 次近邻描述符的匹配。
# 对于 matches 中的每一对 m, n（其中 m 是最近邻匹配，n 是次近邻匹配）：
# 计算 m.distance 和 n.distance 的比值。
# 如果这个比值小于0.5（即最近邻的距离是次近邻距离的一半或更少），那么认
# 为这个匹配是“好”的，并将其添加到 goodMatch 列表中。
# distance：查询描述符和训练描述符之间的距离。这个距离取决于你使用的匹配
# 器（如BFMatcher）和描述符类型。对于使用L2范数的匹配器，这将是欧几里得
# 距离；对于使用汉明距离的二进制描述符（如ORB），这将是汉明距离。
goodMatch = []
for m, n in matchs:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(iphone1, kp1, iphone2, kp2, goodMatch[:20])













# 第二次写

def DrawMatchesKnn_cv2(iphone1, kp1, iphone2, kp2, goodMatch):
    high1, width1 = iphone1.shape[:2]
    high2, width2 = iphone2.shape[:2]
    # 这里的参数np.uint8一定要设置，否则无法正确显示图像
    vis = np.zeros((max(high1, high2), width1 + width2, 3), np.uint8)
    vis[:high1, :width1] = iphone1
    vis[:high2, width1:width1 + width2] = iphone2

    queryIdx = [DMatch.queryIdx for DMatch in goodMatch]
    trainIdx = [DMatch.trainIdx for DMatch in goodMatch]

    post1 = np.int32([kp1[idx].pt for idx in queryIdx])
    # 要偏移
    post2 = np.int32([kp2[idx].pt for idx in trainIdx]) + (width1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), color=(0, 0, 255))
    cv2.imshow("vis", vis)
    cv2.waitKey(0)

    pass


# 读取图片
iphone1 = cv2.imread("iphone1.png")
iphone2 = cv2.imread("iphone2.png")

# 创建sift对象
sift = cv2.SIFT_create()
# 特征点检测
kp1, descriptors1 = sift.detectAndCompute(iphone1, None)
kp2, descriptors2 = sift.detectAndCompute(iphone2, None)

# 特征点匹配，创建匹配器，使用二范数作为特征点匹配的度量，即欧氏距离
BFMatcher = cv2.BFMatcher(cv2.NORM_L2)

matchs = BFMatcher.knnMatch(descriptors1, descriptors2, k=2)
goodMatch = []
for m, n in matchs:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

DrawMatchesKnn_cv2(iphone1, kp1, iphone2, kp2, goodMatch)

# iphone1=cv2.drawKeypoints(iphone1,kp1,iphone1,color=(100,123,168),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# iphone2=cv2.drawKeypoints(iphone2,kp2,iphone2,color=(100,123,168),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("iphone1",iphone1)
# cv2.imshow("iphone2",iphone2)
# cv2.waitKey(0)
