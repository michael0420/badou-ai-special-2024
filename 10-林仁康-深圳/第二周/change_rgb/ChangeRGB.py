'''
@author linrenkang
将图像转为灰度图 & 二值化
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 灰度范围
GRAY_RANGE = 255

# 画图的规模
SCALE = 2
# 画图排序
SRC_IDX = 1
GRAY_IDX = 2
BIN_IDX = 3

class MyImg:
    # 原图像
    img = None
    # 灰度图
    grayImg = None
    # 二值化图
    binImg = None

    def __init__(self, path):
        self.img = cv2.imread(path)

    # ----------------- 私有函数 -------------
    # 灰度算法
    def ToGrayAlgorithm(self, imgUnit):
        return 0.11 * imgUnit[0] + 0.59 * imgUnit[1] + 0.33 * imgUnit[2]

    # 二值化算法
    def ToBinAlgorithm(self, imgUnit):
        if imgUnit <= 0.5:
            return 0
        return 1

    # 清空灰度矩阵
    def ClearGrayImg(self):
        self.grayImg = np.zeros([self.img.shape[0], self.img.shape[1]], self.img.dtype)

    # 清空二值矩阵
    def ClearBinImg(self):
        self.binImg = np.zeros([self.img.shape[0], self.img.shape[1]], self.img.dtype)

    # 清理二值化的输入
    def CleanBinInput(self):
        return self.grayImg / GRAY_RANGE

    # 将原图像转为目的图像
    def BaseFunc(self, func, srcImg, dstImg):
        h, w = self.img.shape[:2]
        for i in range(h):
            for j in range(w):
                dstImg[i, j] = func(srcImg[i, j])

    # 输出图片
    def ShowPic(self, img, idx):
        print("Picture", img)
        print("Shape", img.shape)
        imgTmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(SCALE, SCALE, idx)
        plt.imshow(imgTmp)

    # 输出灰度图片
    def ShowGrayPic(self, img, idx):
        print("Picture", img)
        print("Shape", img.shape)
        plt.subplot(SCALE, SCALE, idx)
        plt.imshow(img, cmap='gray')

    # ----------------- 导出接口 -------------
    # 计算灰度矩阵
    def ToGray(self):
        self.ClearGrayImg()
        self.BaseFunc(self.ToGrayAlgorithm, self.img, self.grayImg)
        print("ok grep: ", self.grayImg)

    # 计算二值化矩阵
    def ToBin(self):
        self.ClearBinImg()
        inputGray = self.CleanBinInput()
        self.BaseFunc(self.ToBinAlgorithm, inputGray, self.binImg)

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img, SRC_IDX)

    # 输出灰度图像
    def ShowGrayImg(self):
        self.ShowGrayPic(self.grayImg, GRAY_IDX)

    # 输出二值化图像
    def ShowBinImg(self):
        self.ShowGrayPic(self.binImg, BIN_IDX)

    # 展示画布
    def Display(self):
        plt.show()

'''
以下为灰度图 & 二值化的应用
'''

# 原图
myImg = MyImg("../pic/lenna.png")
myImg.ShowSrcImg()

# 灰度化
myImg.ToGray()
myImg.ShowGrayImg()

# 二值化
myImg.ToBin()
myImg.ShowBinImg()

# 展示画布
myImg.Display()

