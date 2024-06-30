
'''
@author linrenkang
KMeans 实现
'''
import copy
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class MyData:
    # 源数据
    srcD = None

    # k 簇
    k = 0
    # 质心
    centroid = []
    # 数据分类
    calD = []
    # 最大尝试次数
    maxTry = 0

    # debug 开关
    debugEnable = False

    # 绘图
    xList = []
    xList = []
    pointCluster = []

    def __init__(self, srcD):
        self.srcD = srcD
        self.xList = [point[1] for point in srcD]
        self.yList = [point[0] for point in srcD]
        self.pointCluster = [0 for _ in range(len(srcD))]

    def _Draw(self):
        plt.title("Kmeans-Basketball Data")
        plt.ylabel("assists_per_minute")
        plt.xlabel("points_per_minute")
        # plt.scatter(self.xList, self.yList, c=self.pointCluster)
        plt.scatter(self.yList, self.xList, c=self.pointCluster)
        plt.show()

    # ------------ 私有函数 -------------
    # 初始化随机质点
    def _GetInitCentroid(self):
        # 初始化质点
        randomList = []
        num = 0
        while num < self.k:
            point = random.randint(0, len(self.srcD) - 1)
            if point in randomList:
                continue
            self.centroid.append(self.srcD[point])
            num += 1

    # 初始化
    def _Init(self, k):
        self.k = k
        self.calD = [[] for _ in range(k)]
        self.centroid.clear()
        self._GetInitCentroid()
        self._Debug("debug: 初始化质点 ", self.centroid)

    def _Debug(self, *args):
        if not self.debugEnable:
            return
        print(*args)

    # 检查质心是否相等
    def _CheckCentroid(self, src, dst):
        np.sort(src)
        np.sort(dst)

        # 检查是否横纵坐标都相等
        for i in range(len(src)):
            if not (src[i][0] == dst[i][0] and src[i][1] == dst[i][1]):
                return False
        return True

    # 计算距离
    def _CalDistance(self, srcX, srcY, dstX, dstY):
        return (dstX - srcX) ** 2 + (dstY - srcY) ** 2

    # 计算最近的簇
    def _GetShortCluster(self, centroid, x, y):
        idx = 0
        distance = self._CalDistance(x, y, centroid[0][1], centroid[0][0])

        # 遍历获取距离最近的点
        for i in range(len(centroid)):
            distanceTmp = self._CalDistance(x, y, centroid[i][1], centroid[i][0])
            if distanceTmp < distance:
                idx = i
                distance = distanceTmp
        return idx

    # 计算新一轮质心
    def _CalNewCentroid(self, pointSet):
        return np.mean(pointSet, axis=0)

    # 实现K-means 核心算法
    def _KMeanAlgorithm(self, src, cnt):
        # 达到最大尝试次数，则结束
        if self.maxTry != 0 and cnt >= self.maxTry:
            return True
        self._Debug("debug: 尝试次数: ", cnt)
        self._Debug("debug: 质心", src)

        # 清空各个分类
        for i in range(self.k):
            self.calD[i].clear()

        # 开始分类
        c = 0
        for y, x in self.srcD:
            cluster = self._GetShortCluster(src, x, y)
            self.calD[cluster].append([y, x])
            self.pointCluster[c] = cluster
            c += 1
        self._Debug("debug: 各个簇分类", self.calD)
        self._Debug("debug: 每个坐标属于簇", self.pointCluster)

        # 计算新一轮质心
        dst = []
        for i in range(self.k):
            if len(self.calD[i]) == 0:
                return False
            y, x = self._CalNewCentroid(self.calD[i])
            dst.append([y, x])
        self._Debug("debug: 新一轮质心", dst)

        # 质心有变化，则继续算法
        if not self._CheckCentroid(src, dst):
            return self._KMeanAlgorithm(dst, cnt + 1)
        return True

    # 输出图片
    def _ShowPic(self, img):
        self._Debug("Picture: ", img)
        self._Debug("Shape: ", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出图片
    def _ShowPicName(self, img, name):
        self._Debug("Picture: " + name, img)
        self._Debug("Shape: ", img.shape)
        cv2.imshow("Picture: " + name, img)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def _ShowMergePic(self, imgSrc, chImg):
        self.ShowPic(chImg)
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------

    # K-means 算法
    def kMeansDetail(self, k, maxTry=0):
        # 防止因初始化质点随机性，有可能得不到 k 类，重做来处理
        while True:
            # 1. 初始化(设置K和初始化质点)
            self._Init(k)
            self.maxTry = maxTry
            # 2. 根据质点实现K-means
            if self._KMeanAlgorithm(self.centroid, 0):
                break
        # 3. 图像显示
        self._Draw()

    def kMeans(self, k):
        clf = KMeans(n_clusters=k)
        self.pointCluster = clf.fit_predict(self.srcD)
        self._Draw()

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像（彩色图）
    def ShowDstImg(self):
        self.ShowMergePic(self.img, self.chImg)

'''
以下为 Kmeans 应用
'''

if __name__ == "__main__":
    """
    第一部分：数据集
    X表示二维矩阵数据，篮球运动员比赛数据
    总共20行，每行两列数据
    第一列表示球员每分钟助攻数：assists_per_minute
    第二列表示球员每分钟得分数：points_per_minute
    """
    X = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
    ]
    myData = MyData(X)

    # 本地实现Kmeans
    myData.kMeansDetail(3)
    # 调用接口实现Keans
    myData.kMeans(3)
