'''
@author linrenkang
使用邻近插值法实现图像缩放
'''

import cv2
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 缩放后的图像
    chImg = None

    # 目标 size
    dstHeight = 0
    dstWidth = 0

    # 行和宽缩放比例
    hScale = 1
    wScale = 1

    def __init__(self, path):
        self.img = cv2.imread(path)

    # ------------ 私有函数 -------------
    # 设置目标 size
    def SetSize(self, dstH, dstW):
        self.dstHeight = dstH
        self.dstWidth = dstW

    # 设置缩放比例
    def SetScale(self, hScale, wScale):
        self.hScale = hScale
        self.wScale = wScale

    # 清空转换图像
    def ClearChImg(self):
        self.chImg = np.zeros([self.dstHeight, self.dstWidth, self.img.shape[2]], self.img.dtype)

    # 越界处理
    def HandleCross(self, x, y):
        h, w = self.img.shape[:2]
        if y >= h:
            y = h - 1
        if x >= w:
            x = w - 1
        return x, y

    # 使用最邻近插值
    def ExtendByAdjacentPoint(self):
        self.ClearChImg()
        for i in range(self.dstHeight):
            for j in range(self.dstWidth):
                adjacentY = round(i / self.hScale)
                adjacentX = round(j / self.wScale)
                adjacentX, adjacentY = self.HandleCross(adjacentX, adjacentY)
                self.chImg[i, j] = self.img[adjacentY, adjacentX]

    # 输出图片
    def ShowPic(self, img):
        print("Picture", img)
        print("Shape", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 根据目标规模缩放
    def SamplingBySpecNum(self, height, width):
        h, w = self.img.shape[:2]
        self.SetScale(height / h, width / h)
        self.SetSize(height, width)
        self.ExtendByAdjacentPoint()

    # 指定比例缩放
    def SamplingBySpecScale(self, scale):
        h, w = self.img.shape[:2]
        self.SetScale(scale, scale)
        self.SetSize(round(h * scale), round(w * scale))
        self.ExtendByAdjacentPoint()

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像
    def ShowDstImg(self):
        self.ShowPic(self.chImg)

'''
以下为邻近插值法的应用
'''
myImg = MyImg('../pic/lenna.png')
# 输出原图
myImg.ShowSrcImg()

# 指定 800 * 800 大小放大
myImg.SamplingBySpecNum(800, 800)
myImg.ShowDstImg()
#
# # 指定 80 * 80 缩小
myImg.SamplingBySpecNum(80, 80)
myImg.ShowDstImg()

# 放大为原来 2 倍
myImg.SamplingBySpecScale(2)
myImg.ShowDstImg()

# # 缩小为原来 1/2
myImg.SamplingBySpecScale(0.5)
myImg.ShowDstImg()
