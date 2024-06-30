'''
@author linrenkang
标准化实现
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

class MyData:
    srcX = None
    srcY = None
    dstX = None

    # 辅助张量优化计算效率
    mean = 0
    min = 0
    max = 0
    size = 0
    srcM = None
    dstM = None
    meanM = None
    minX = None

    # ---- 私有函数 ----
    def __init__(self, srcX, srcY):
        self.srcX = srcX
        self.srcY = srcY
        self.mean = np.mean(srcX)
        self.min = min(srcX)
        self.max = max(srcX)
        self.size = len(srcX)
        self.srcM = np.array(srcX)
        self.meanM = np.array([self.mean] * self.size)
        self.minM = np.array([self.min] * self.size)

    # ---- 导出函数 ----
    # 范围集中在[-1, 1]
    def SetStand0(self):
        self.dstM = (self.srcM - self.minM) / (self.max - self.min)

    # 范围集中在[0, 1]
    def SetStand1(self):
        self.dstM = (self.srcM - self.minM) / (self.max - self.min)

    # 正太分布
    def SetStandZcore(self):
        subM = self.srcM - self.meanM
        s2 = sum(subM * subM) / self.size
        self.dstM = (self.srcM - self.meanM) / s2

    # 正太分布
    def SetStandZcoreIntf(self):
        self.dstM = np.array(zscore(srcX))

    # 绘图
    def ShowPic(self):
        self.dstX = list(self.dstM)
        plt.plot(self.srcX, self.srcY)
        plt.plot(self.dstX, self.srcY)
        plt.show()

'''
@author linrenkang
标准化应用
'''
if __name__ == "__main__":
    srcX = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    # 个数简单作为 Y 轴
    srcY = []
    for e in srcX:
        srcY.append(srcX.count(e))

    # 初始化
    myData = MyData(srcX, srcY)

    # 标准化在 [-1, 1]
    myData.SetStand0()
    myData.ShowPic()

    # 标准化在 [0, 1]
    myData.SetStand1()
    myData.ShowPic()

    # 使用正太分布标准化
    myData.SetStandZcore()
    myData.ShowPic()

    # 使用正太分布标准化（调用接口）
    myData.SetStandZcoreIntf()
    myData.ShowPic()