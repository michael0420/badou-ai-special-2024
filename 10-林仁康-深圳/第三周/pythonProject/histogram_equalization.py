'''
@author linrenkang
直方图均衡化
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 灰度图
    grayImg = None
    # 缩放后的图像
    chImg = None

    # 像素极大值
    PIXEL = 256

    def __init__(self, path):
        self.img = cv2.imread(path)

    # ------------ 私有函数 -------------
    # 展示直方图
    def ShowHist(self):
        plt.figure()
        if len(self.chImg.shape) <= 2:
            # ravel() 函数将数组多维度拉成一维数组
            plt.hist(self.chImg.ravel(), self.PIXEL)
        else:
            b, g, r = cv2.split(self.chImg)
            plt.hist(b.ravel(), self.PIXEL)
            plt.hist(g.ravel(), self.PIXEL)
            plt.hist(r.ravel(), self.PIXEL)
        plt.show()

    # 输出图片
    def ShowPic(self, img):
        print("Picture", img)
        print("Shape", img)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def ShowMergePic(self, imgSrc, chImg):
        self.ShowPic(chImg)
        self.ShowHist()
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 根据目标规模缩放
    def GrayHistEq(self):
        # 灰度化
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        self.chImg = cv2.equalizeHist(self.grayImg)

    def RGBHistEq(self):
        # 按通道拆分
        b, g, r = cv2.split(self.img)

        # 直方图均衡化
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)

        # 合并图像
        self.chImg = cv2.merge((b, g, r))

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像（灰度图）
    def ShowDstImgByGray(self):
        self.ShowMergePic(self.grayImg, self.chImg)

    # 输出转换后的图像（彩色图）
    def ShowDstImgByRGB(self):
        self.ShowMergePic(self.img, self.chImg)

'''
以下为直方图均衡化的应用
'''
myImg = MyImg('pic/lenna.png')
# 输出原图
myImg.ShowSrcImg()

# 灰度直方图均衡化
myImg.GrayHistEq()
myImg.ShowDstImgByGray()

# RGB直方图均衡化
myImg.RGBHistEq()
myImg.ShowDstImgByRGB()
