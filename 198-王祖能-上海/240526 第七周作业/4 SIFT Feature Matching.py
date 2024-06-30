'''
图像相似度，特征匹配
'''
import cv2
import numpy as np


def drawMatchesKnn_Manual(src1, kp1, src2, kp2, good_match):
    h1, w1 = src1.shape[:2]
    h2, w2 = src2.shape[:2]
    merge = np.zeros([max(h1, h2), w1 + w2, 3], dtype=np.uint8)
    merge[:h1, :w1] = src1
    merge[:h2, w1:w1 + w2] = src2

    Idx1 = [kpp.queryIdx for kpp in good_match]
    Idx2 = [kpp.trainIdx for kpp in good_match]
    Distance = [kpp.distance for kpp in good_match]

    post1 = np.int16([kp1[pp].pt for pp in Idx1])  # 对应序号的特征点坐标，并归整
    post2 = np.int16([kp2[pp].pt for pp in Idx2]) + [w1, 0]
    print(post1, post2)
    print(Distance)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.circle(merge, (x1, y1), radius=5, color=(0, 255, 0))
        cv2.circle(merge, (x2, y2), radius=5, color=(0, 255, 0))
        cv2.line(merge, (x1, y1), (x2, y2), color=(0, 255, 0))

    cv2.imshow('feature matching', merge)
    cv2.waitKey()
    cv2.destroyAllWindows()


img1, img2 = cv2.imread('iphone1.png'), cv2.imread('iphone2.png')
sift = cv2.SIFT_create()
kps1, des1 = sift.detectAndCompute(img1, mask=None)
kps2, des2 = sift.detectAndCompute(img2, mask=None)
print(kps1[0].pt)  # keypoint类变量，pt表示特征点坐标，size邻域直径，angle特征点方向。。。
# brute force 暴力匹配 创建实例对象，并使用对象方法
bf = cv2.BFMatcher(cv2.NORM_L2, False)
# DMatch数据结构包含三个非常重要的数据分别是queryIdx(某一特征点在本帧图像的索引)，trainIdx(该特征点在另一张图像中相匹配的特征点的索引)，distance(距离）
# 假如特征点A使用暴力匹配到的特征点B；反向特征点B进行匹配的仍是特征点A，则是正确的匹配，否则错误。
'''
第一个参数normType用来指定距离测试类型。默认cv2.Norm_L2,适合SIFT和SURF等。（c2.NORM_L1 也可以）
第二个参数crossCheck布尔变量 ，默认值为False。如果为True，必须距离最近方可匹配
创建的BFMatcher 对象两个方法:
BFMatcher.match() 会返回最佳匹配。
BFMatcher.knnMatch(self, queryDescriptors, trainDescriptors, k, mask=None, compactResult=None) 为每个关键点返回 k 个最佳匹配（降序后前 k个）,除了匹配之外还要做其他事情（比如进行比值测试）。
'''
matches = bf.knnMatch(des1, des2, k=2)
# knnMatch匹配的返回结果是一个元组：说明结果不能改变；knnMatch与match的返回值类型一样，只不过一组返回的2个DMatch类型，返回最匹配的两个点的DMatch信息
# 找到距离最近的两个点，三通道像素值平方和开方，(b1-b2）^2+(g1-g2)^2+(r1-r2)^2后开方, K nearest neighbor
good_matches = []
for m, n in matches:
    print(m.distance, n.distance)
    '''
    DMatch.distance：描述符之间的距离。越小越好。DMatch.trainIdx ： 目标图像中描述符的索引。
    DMatch.queryIdx ：查询图像中描述符的索引。
    DMatch.trainIdx ：目标图像的索引。
    '''
    if m.distance < 0.35 * n.distance:  # 某A点的两个匹配点B/C，到B的距离如果是到C距离的一半，认为区别足够大，选择最近的作为匹配点
        good_matches.append(m)
    print(m.queryIdx, m.trainIdx)
drawMatchesKnn_Manual(img1, kps1, img2, kps2, good_matches)

# cv2.drawMatchesKnn(img1, kps1, img2, kps2, matches[:4], None, flags=2)  # 绘图是否带knn要看上一句是否是knn匹配,K-NearestNeighbor是k个最邻近匹配
