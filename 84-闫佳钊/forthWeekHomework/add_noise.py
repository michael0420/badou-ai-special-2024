'''
增加噪声：高斯噪声和椒盐噪声
'''
import random
import cv2
import numpy as np


class AddNoise:
    imgSrc = None  # 原图像
    imgDst = None  # 目标图像
    coverRate = 0  # 信噪比
    noiseCnt = 0  # 噪声数量

    # 高斯分布
    mean = 0
    sigma = 0

    def __init__(self, path):
        self.imgSrc = cv2.imread(path)
        self.imgSrc = cv2.cvtColor(self.imgSrc, cv2.COLOR_BGR2GRAY)

    def _initImgDst(self):
        self.imgDst = cv2.copyTo(self.imgSrc, self.imgDst)

    # 高斯噪声点
    def _GaussionNoiseAlgorithm(self, x, y):
        return self.imgDst[y][x] + random.gauss(self.mean, self.sigma)

    # 椒盐噪声点
    def _SpicySaltAlgorithm(self, x, y):
        if (random.random() < 0.5):
            return 0
        return 255

    # 生成噪声点的位置
    def _GenRandomPoint(self):
        h, w = self.imgSrc.shape[:2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        return x, y

    # 检查像素值是否合理，并修正
    def _CheckVal(self, val):
        if val < 0:
            return 0
        if val > 255:
            return 255
        return int(val)

    # 产生噪声
    def _GenNoise(self, func):
        for i in range(self.noiseCnt):
            x, y = self._GenRandomPoint()
            self.imgDst[y][x] = self._CheckVal(func(x, y))

    # 输出原始图片
    def _ShowPic(self, img):
        print("Picture", img)
        print("Shape", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出原图与目标图片
    def _ShowMergePic(self, imgSrc, imgDst):
        cv2.imshow("Picture", np.hstack([imgSrc, imgDst]))
        cv2.waitKey(0)

    # 使用高斯分布产生噪声点（灰度）
    def CaussionNoise(self, mean, sigma, rate):
        h, w = self.imgSrc.shape[:2]
        self.mean = mean
        self.sigma = sigma
        self.noiseCnt = int(h * w * rate)
        self._initImgDst()
        self._GenNoise(self._GaussionNoiseAlgorithm)

    # 产生椒盐噪声点
    def SpicySaltNoise(self, rate):
        h, w = self.imgSrc.shape[:2]
        self.noiseCnt = int(h * w * rate)
        self._initImgDst()
        self._GenNoise(self._SpicySaltAlgorithm())

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.imgSrc)

    # 输出转换后的图像
    def ShowDstImg(self):
        self._ShowMergePic(self.imgSrc, self.imgDst)


# 实际应用
if __name__ == "__main__":
    img = AddNoise('lenna.png')
    # 在灰度图基础上，生成随机高斯噪声
    img.CaussionNoise(2, 4, 0.6)
    img.ShowDstImg()
    # 在灰度图基础上，生成随机椒盐噪声
    img.SpicySaltNoise(0.3)
    img.ShowDstImg()
