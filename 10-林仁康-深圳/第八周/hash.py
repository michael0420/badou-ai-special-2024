'''
@author linrenkang
hash 算法，图像指纹
'''
import random

import cv2
import numpy as np

class MyImg:
    # 源数据
    srcImg = None

    # debug
    debugEnable = True

    def __init__(self, path):
        self.srcImg = cv2.imread(path)

    # ------------ 私有函数 -------------
    def _Debug(self, *args):
        if not self.debugEnable:
            return
        print(*args)

    def _ShowPic(self, img):
        print("Picture", img)
        print("Shape", img)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 均值 hash 核心算法
    def _MeamHashAlgorithm(self, sizeWH, imgGray):
        # 1 计算均值
        meanNum = np.mean(imgGray)
        # 2 计算 hash 矩阵
        return np.mat(imgGray) - meanNum

    # 差值 hash 核心算法
    def _DValHashAlgorithm(self, sizeWH, imgGray):
        # 1 获取计算的矩阵
        col = sizeWH[0]
        mat = np.mat(imgGray)
        A = mat[:, :-1].astype(np.int32)
        B = mat[:, 1:].astype(np.int32)
        # 2 计算 hash 矩阵
        return A - B

    # hash 矩阵转换 hash 值
    def _HashMatToStr(self, hashMat):
        hashStr = ""
        for i in range(len(hashMat)):
            for j in range(len(hashMat[:, 0])):
                if hashMat[i, j] > 0:
                    hashStr += "1"
                else:
                    hashStr += "0"
        return hashStr

    # hash 核心框架
    def _Hash(self, sizeWH, algorithm):
        # 1 缩放
        imgResize = cv2.resize(self.srcImg, sizeWH, interpolation=cv2.INTER_LINEAR)
        # 2 灰度图
        imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
        # 3 计算 hash 矩阵
        hashMat = algorithm(sizeWH, imgGray)
        # 4 计算 hash 值
        return self._HashMatToStr(hashMat)

    #----------------- 导出接口 -------------
    # 均值 hash
    def MeanHash64(self):
        hashStr = self._Hash((8, 8), self._MeamHashAlgorithm)
        print("Mean hash", hashStr)

    # 差值 hash
    def DValHash64(self):
        hashStr = self._Hash((9, 8), self._DValHashAlgorithm)
        print("DVal hash", hashStr)

    # 输出图像
    def ShowSrcImg(self):
        self._ShowPic(self.srcImg)

'''
以下为 hash 的应用
'''

if __name__ == "__main__":
    myImg = MyImg("pic/lenna.png")
    # myImg.ShowSrcImg()

    # 均值 hash
    myImg.MeanHash64()

    # 差值 hash
    myImg.DValHash64()
