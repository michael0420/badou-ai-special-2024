'''
@author linrenkang
sobel 边缘检测
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 水平方向检测
    xChImg = None
    # 垂直方向检测
    yChImg = None
    # 目标图像
    chImg = None

    # 像素极大值
    PIXEL = 256

    def __init__(self, path):
        self.img = cv2.imread(path)

    # ------------ 私有函数 -------------
    # 输出图片
    def ShowPic(self, img):
        print("Picture", img)
        print("Shape", img)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 水平方向边缘检测
    def SobelX(self):
        # 使用 sobel 边缘在水平方向检测
        tmp = cv2.Sobel(self.img, cv2.CV_16S, 1, 0)
        # 修正像素，支持图片输出
        self.xChImg = cv2.convertScaleAbs(tmp)

    # 垂直方向边缘检测
    def SobelY(self):
        # 使用 sobel 边缘在垂直方向检测
        tmp = cv2.Sobel(self.img, cv2.CV_16S, 0, 1)
        # 修正像素，支持图片输出
        self.yChImg = cv2.convertScaleAbs(tmp)

    #----------------- 导出接口 -------------
    # Sobel 边缘检测
    def Sobel(self):
        self.SobelX()
        self.SobelY()
        self.chImg = cv2.addWeighted(self.xChImg, 0.5, self.yChImg, 0.5, 0)

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像（灰度图）
    def ShowDstImg(self):
        self.ShowPic(self.xChImg)
        self.ShowPic(self.yChImg)
        cv2.imshow("Picture", np.hstack([self.xChImg, self.yChImg, self.chImg]))
        cv2.waitKey(0)

'''
以下sobel的应用
'''
myImg = MyImg('pic/lenna.png')
# 输出原图
myImg.ShowSrcImg()

# 灰度直方图均衡化
myImg.Sobel()
myImg.ShowDstImg()
