'''
@author linrenkang
线性回归：最小二乘法、RANSAC
'''
import random

import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl

def GenTestData(IsolationEnable=True):
    # 生成理想数据
    nSamples = 500  # 样本个数
    nInputs = 1  # 输入变量个数
    nOutputs = 1  # 输出变量个数
    # np.random.random产生范围0-1之间，随机生成0-20之间的500个数据:行向量
    XExact = 20 * np.random.random((nSamples, nInputs))
    perfectFit = 60 * np.random.normal(size=(nInputs, nOutputs))  # 随机线性度，即随机生成一个斜率
    YExact = np.dot(XExact, perfectFit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    XNoisy = XExact + np.random.normal(size=XExact.shape)  # 500 * 1行向量,代表Xi
    YNoisy = YExact + np.random.normal(size=YExact.shape)  # 500 * 1行向量,代表Yi

    if IsolationEnable:
        # 添加"局外点"
        nOutliers = 100
        allIdxs = np.arange(XNoisy.shape[0])  # 获取索引0-499
        # 基于 allIdxs 序列进行打乱，随机选取其中的点，变为局外点
        np.random.shuffle(allIdxs)
        outlierIdxs = allIdxs[:nOutliers]
        # 产生 nOutliers 个局外点，并覆盖原有的点
        XNoisy[outlierIdxs] = 20 * np.random.random((nOutliers, nInputs))
        YNoisy[outlierIdxs] = 50 * np.random.normal(size=(nOutliers, nOutputs))

    # return [XNoisy, YNoisy]
    return np.concatenate((XNoisy, YNoisy), axis=1), np.concatenate((XExact, YExact), axis=1)

class MyData:
    # 源数据
    srcD = None
    exactD = None
    bestiInliers = None

    # 最小二乘法结果
    k = 0
    b = 0
    ASL = None
    ARansac = None
    YLSFit = None
    YRansac = None

    # RANSAC 参数
    ranInnerNum = 0
    maxTry = 0

    # debug
    debugEnable = True

    def __init__(self, srcD, exactD=None):
        self.srcD = srcD
        self.exactD = exactD

    # ------------ 私有函数 -------------
    def _Debug(self, *args):
        if not self.debugEnable:
            return
        print(*args)

    # 绘制散点图
    def _DrawScatter(self, X, Y, mk='k.', lb='data'):
        pylab.plot(X, Y, mk, label=lb)  # 散点图
        pylab.legend()

    # 绘制线条
    def _DrawLine(self, X, Y, lb='data'):
        pylab.plot(X, Y, label=lb)  # 散点图
        pylab.legend()

    # 绘制画布
    def _ShowPic(self):
        pylab.show()

    # 单个值转换标量计算
    def _ToScalar(self, matrix):
        return matrix[0, 0]

    # 最小二乘核心算法
    def _LeastSquare(self, data):
        srcD = data
        n = len(srcD)
        sumX = np.sum(srcD[:, 0])
        sumY = np.sum(srcD[:, 1])
        sumXX = self._ToScalar(srcD[:, 0].T * srcD[:, 0])
        sumXY = self._ToScalar(srcD[:, 1].T * srcD[:, 0])
        denominator = n * sumXX - sumX * sumX

        k = (n * sumXY - sumX * sumY) / denominator
        b = (sumY * sumXX - sumX * sumXY) / denominator
        return k, b

    # 计算线性结果
    def _Linear(self, k, b, X):
        return np.multiply(X, k) + b

    # 检测是否为内群
    def _GetInnerSet(self, srcD, Yh, deviation):
        YSub = np.array(Yh - srcD[:, 1])
        # 向量积
        YSqrSum = YSub * YSub
        return srcD[np.array(YSqrSum).flatten() < deviation]

    # 检测是否为更优的参数
    def _CheckBestInliers(self, bestSet, newSet, leastNum):
        # return len(newSet) >= leastNum and len(newSet) > len(bestSet)
        return len(newSet) > len(bestSet)

    # 最小二乘法核心算法（线性代数）
    def _LeastSquare2(self):
        # 训练数据
        srcD = np.mat(self.srcD)
        X = srcD[:, 0]
        Y = srcD[:, 1]

        # 线性代数计算
        # 在 in SciPy 2.0.0 已经抛弃
        # A, resids, rank, s = sl.lstsq(X, Y)
        A, resids, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        self._Debug("解", A)
        self._Debug("残差和（实际值与估计值之差的平方的总和）", resids)
        self._Debug("矩阵 X 的秩", rank)
        self._Debug("奇异值（分解对角矩阵上的值）", s)
        self._Debug("验证值", "X=", X, "Y=", self.YLSFit)
        return A

    # RANSAC 核心算法
    def _Ransac2Core(self, ranInnerNum, maxTry, deviation, leastNum):
        inliersBest = None
        ABest = None
        for i in range(maxTry):
            # 1. 随机挑选序列
            choiceRanSetIdx = random.sample(range(len(self.srcD)), ranInnerNum)
            choiceRanSet = self.srcD[choiceRanSetIdx]
            # 2. 使用拟合模型计算参数
            XRan = np.mat(choiceRanSet)[:, 0]
            YRan = np.mat(choiceRanSet)[:, 1]
            A, _, _, _ = np.linalg.lstsq(XRan, YRan, rcond=None)
            # 3 计算拟合直线
            srcD = np.mat(self.srcD)
            X = srcD[:, 0]
            Yh = np.dot(X, A)
            # 4. 计算内群
            inliers = self._GetInnerSet(srcD, Yh, deviation)
            print("inliers len", len(inliers))
            # 5. 挑选最优拟合集合和参数
            if inliersBest is None or self._CheckBestInliers(inliersBest, inliers, leastNum):
                inliersBest, ABest = inliers, A
        return ABest, inliersBest

    #----------------- 导出接口 -------------
    # 最小二乘法
    def LeastSquare(self):
        self.k, self.b = self._LeastSquare(self.srcD)
        self._Debug("最小二乘法 k, b", self.k, self.b)

    # 最小二乘法（线性代数计算）
    def LeastSquareInteraface(self):
        # 原始数据
        Xexact = np.mat(self.exactD)[:, 0]
        self.ALS = self._LeastSquare2()
        self.YLSFit = np.dot(Xexact, self.ALS)

    # RANSAC 算法
    def Ransac(self, ranInnerNum, maxTry, deviation, leastNum=300):
        inliersBest = None
        kBest = None
        bBest = None
        for i in range(maxTry):
            # 1 随机挑选序列
            choiceRanSetIdx = random.sample(range(len(self.srcD)), ranInnerNum)
            choiceRanSet = self.srcD[choiceRanSetIdx]
            # 2 使用拟合模型计算参数，计算拟合集合
            k, b = self._LeastSquare(choiceRanSet)
            Yh = self._Linear(k, b, self.srcD[:, 0])
            inliers = self._GetInnerSet(self.srcD, Yh, deviation)
            # 3 挑选最优拟合集合和参数
            if inliersBest is None or self._CheckBestInliers(inliersBest, inliers, leastNum):
                inliersBest, kBest, bBest = inliers, k, b
        self.k, self.b, self.inliersBest = kBest, bBest, inliersBest
        self._Debug("RANSAC k, b", self.k, self.b)

    # RANSAC 算法（线性代数）
    def Ransac2(self, ranInnerNum, maxTry, deviation, leastNum=300):
        # 原始数据
        Xexact = np.mat(self.exactD)[:, 0]

        # RANSAC 核心算法
        self.ARansac, self.inliersBest = self._Ransac2Core(ranInnerNum, maxTry, deviation, leastNum)
        self.YRansac =  np.dot(Xexact, self.ARansac)

    def ShowData(self):
        exactD = np.mat(self.exactD)
        # 原始数据
        Xexact = exactD[:, 0]
        Yexact = exactD[:, 1]

        # 噪声数据
        XNoise = self.srcD[:, 0]
        YNoise = self.srcD[:, 1]

        # RANSAC
        XRansc = np.mat(self.inliersBest)[:, 0]
        YRansc = np.mat(self.inliersBest)[:, 1]

        # 全部散点
        self._DrawScatter(XNoise, YNoise, lb='data')
        # 实际正确直线
        self._DrawLine(Xexact, Yexact, 'exact system')
        # 最小二乘法拟合直线
        self._DrawLine(Xexact, self.YLSFit, lb='linear fit')
        # RANSA 拟合直线
        self._DrawScatter(XRansc, YRansc, mk='bx', lb='RANSAC')
        self._DrawLine(Xexact, self.YRansac, lb='ransac fit')
        self._ShowPic()

'''
以下为最小二乘法、RANSAC的应用
'''

if __name__ == "__main__":
    # srcD = np.mat([[1, 6], [2, 5], [3, 7], [4, 10]])
    srcD, exactD = GenTestData()
    myData = MyData(srcD, exactD)

    # 最小二乘法(线性代数)
    myData.LeastSquareInteraface()

    # RANSAC(线性代数)
    myData.Ransac2(50, 1000, 300)
    myData.ShowData()
