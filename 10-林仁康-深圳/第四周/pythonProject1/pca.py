'''
@author linrenkang
实现 PCA
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class MyData:
    # 原数据
    src = None
    # 标签
    label = None
    # 目标数据
    dst = None
    # 选取降到取 K 个特征值
    K = 0

    # 中间数据
    # 中心化矩阵
    center = None
    # cov 矩阵
    covM = None
    # 特征值
    featureValM = None
    # 特征向量
    featureVecM = None

    # 画图
    redLineX, redLineY = [], []
    greenLineX, greenLineY = [], []
    blueLineX, blueLineY = [], []

    def __init__(self, src):
        self.src = src

    # ------------ 私有函数 -------------
    # 获取中心化
    def _Centralization(self):
        self.center = self.src - self.src.mean()
        print("中心化矩阵：\n", self.center)

    # cov 矩阵:D = (ZT * Z) / (m - 1)
    def _SetCovMatrix(self):
        m = self.center.shape[0]
        Z = self.center
        self.covM = np.dot(Z.T, Z) / (m - 1)
        print("cov 矩阵：\n", self.covM)

    # 特征值与特征向量
    def _SetFeatureMatrix(self):
        self.featureValM, self.featureVecM = np.linalg.eig(self.covM)
        print("特征值：\n", self.featureValM)
        print("特征向量：\n", self.featureVecM)

    # 将特征值大到小排序
    def _SortFeatureIdx(self):
        sortIdx = np.argsort(-self.featureValM)
        print("特征值从大到小排序：\n", sortIdx)
        return sortIdx

    # 特征值筛选特征向量
    def _SortFeatureVec(self, sortIdx):
        sortVec = self.featureVecM[:, sortIdx[:self.K]]
        print("特征向量排序：\n", sortIdx)
        return sortVec

    # 求降维矩阵
    def _SetDstData(self, sortIdx):
        sortFeatureVec = self._SortFeatureVec(sortIdx)
        self.dst = np.dot(self.src, sortFeatureVec)

    # 评估模型好坏
    def _Evaluate(self, sortIdx):
        pickFeatureVal = self.featureValM[sortIdx[:self.K]]
        res = np.sum(pickFeatureVal) / np.sum(self.featureValM)
        print("评估结果：", res)
        return res

    # 使用 sklearn 接口求 PCA 降维矩阵
    def _PCAByInterface(self):
        res = PCA(n_components=self.K)
        res.fit(self.src)
        self.dst = res.fit_transform(self.src)

    # 清除绘图线条
    def _ClearLine(self):
        self.redLineX.clear()
        self.redLineX.clear()

        self.greenLineX.clear()
        self.greenLineY.clear()

        self.blueLineX.clear()
        self.blueLineY.clear()

    # 设置线条
    def _SetTestPicData(self):
        self._ClearLine()
        for i in range(len(self.dst)):
            if self.label[i] == 0:
                self.redLineX.append(self.dst[i][0])
                self.redLineY.append(self.dst[i][1])
                continue
            if self.label[i] == 1:
                self.greenLineX.append(self.dst[i][0])
                self.greenLineY.append(self.dst[i][1])
                continue
            self.blueLineX.append(self.dst[i][0])
            self.blueLineY.append(self.dst[i][1])
        print("红线：\n", self.redLineX, self.redLineY)
        print("绿线：\n", self.greenLineX, self.greenLineY)
        print("蓝线：\n", self.blueLineX, self.blueLineY)

    def _ShowPic(self):
        plt.scatter(self.redLineX, self.redLineY, c='r', marker='X')
        plt.scatter(self.greenLineX, self.greenLineY, c='g', marker='.')
        plt.scatter(self.blueLineX, self.blueLineY, c='b', marker='+')
        plt.show()

    #----------------- 导出接口 -------------
    # 实现 PCA 功能
    def MyPCA(self, K):
        self.K = K
        self._Centralization()
        self._SetCovMatrix()
        self._SetFeatureMatrix()
        sortIdx = self._SortFeatureIdx()
        self._SetDstData(sortIdx)
        self._Evaluate(sortIdx)

    # 调用 sklearn 接口实现 PCA
    def InterfacePCA(self, K):
        self.K = K
        self._PCAByInterface()

    # 鸢尾花测试
    def test(self):
        self.src, self.label = load_iris(return_X_y=True)
        self.InterfacePCA(2)
        self._SetTestPicData()
        self._ShowPic()

    def printSrc(self):
        print("Src Data: \n", self.src)

    def printDst(self):
        print("Dst Data: \n", self.dst)

'''
应用 PCA 提取随机矩阵的特征，与鸢尾花实践
'''
if __name__ == "__main__":
    # X = np.array([[32, 15, 64, 98, 35], [2, 23, 47, 6, 61], [76, 60, 79, 48, 10], [21, 96, 29, 22, 58]])

    # 产生随机数据
    randMinVal = 0
    randMaxVal = 100
    randShape = (4, 5)
    X = np.random.randint(randMinVal, randMaxVal, randShape)

    # PCA
    myData = MyData(X)
    myData.printSrc()

    # 本地实现 PCA
    myData.MyPCA(3)
    myData.printDst()

    # 调用接口 PCA
    myData.InterfacePCA(3)
    myData.printDst()

    # 鸢尾花数据
    myData.test()
    myData.printDst()
