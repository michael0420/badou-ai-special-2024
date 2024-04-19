'''
@author linrenkang
使用二曲性插值实现图像缩放
'''

import cv2
import numpy as np

# 像素类，记录像素的坐标以及值
class Pixel:
    x = 0
    y = 0
    pi = 0

    def __init__(self, x, y, pi):
        self.x = x
        self.y = y
        self.pi = pi

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

    # 平移单位
    MOVE_UINT = 0.5

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

    # 获取对应虚拟点
    def GetPoint(self, dst, scale):
        return (dst + self.MOVE_UINT) / scale - self.MOVE_UINT

    def TwoCurAlgorithm(self, p00, p01, p10, p11, x, y):
        x0 = p00.x
        y0 = p00.y
        x1 = p11.x
        y1 = p11.y
        # 边界时，如果 p00 与 p01 重合，则 f0 为 p00 的像素值
        f0 = p00.pi
        # 同理，如果 p10 与 p11 重合，则 f1 为 p10 的像素值
        f1 = p10.pi
        if x1 != x0:
            f0 = p00.pi * (x1 - x)/(x1 - x0) + p01.pi * (x - x0)/(x1 - x0)
            f1 = p10.pi * (x1 - x)/(x1 - x0) + p11.pi * (x - x0)/(x1 - x0)
        # 边界时，如果 f0 与 f1 重合，则返回 f0 作为像素
        if y1 == y0:
            return f0
        return f0 * (y1 - y)/ (y1 - y0) + f1 * (y - y0)/(y1 - y0)

    # 越界处理
    def HandleCross(self, x, y):
        h, w = self.img.shape[:2]
        return min(x, w - 1), min(y, h - 1)

    # 使用二区性插值法
    def ExtendBy2CurInsert(self):
        if self.hScale == 1 and self.wScale == 1:
            self.chImg = self.img.copy()
        self.ClearChImg()
        for i in range(self.dstHeight):
            srcY = self.GetPoint(i, self.hScale)
            srcY0 = int(srcY)
            srcY1 = int(srcY + 1)
            for j in range(self.dstWidth):
                srcX = self.GetPoint(j, self.wScale)
                srcX0 = int(srcX)
                srcX1 = int(srcX + 1)
                srcX1, srcY1 = self.HandleCross(srcX1, srcY1)
                p00 = Pixel(srcX0, srcY0, self.img[srcY0, srcX0])
                p01 = Pixel(srcX0, srcY1, self.img[srcY0, srcX1])
                p10 = Pixel(srcX1, srcY0, self.img[srcY1, srcX0])
                p11 = Pixel(srcX1, srcY1, self.img[srcY1, srcX1])
                self.chImg[i, j] = self.TwoCurAlgorithm(p00, p01, p10, p11, srcX, srcY)


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
        self.ExtendBy2CurInsert()

    # 指定比例缩放
    def SamplingBySpecScale(self, scale):
        h, w = self.img.shape[:2]
        self.SetScale(scale, scale)
        self.SetSize(round(h * scale), round(w * scale))
        self.ExtendBy2CurInsert()

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像
    def ShowDstImg(self):
        self.ShowPic(self.chImg)

'''
以下为二曲性插值值法的应用
'''
myImg = MyImg('pic/lenna.png')
# 输出原图
myImg.ShowSrcImg()

# 指定 800 * 800 大小放大
myImg.SamplingBySpecNum(800, 800)
myImg.ShowDstImg()

# # 指定 80 * 80 缩小
myImg.SamplingBySpecNum(80, 80)
myImg.ShowDstImg()

# 放大为原来 2 倍
myImg.SamplingBySpecScale(2)
myImg.ShowDstImg()

# # 缩小为原来 1/2
myImg.SamplingBySpecScale(0.5)
myImg.ShowDstImg()
