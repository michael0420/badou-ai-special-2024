import random
import cv2
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 缩放后的图像
    chImg = None
    # 覆盖率
    coverRate = 0
    # 随机位置坐标数组
    noiseCnt = 0
    # 高斯分布
    means = 0
    sigma = 0

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # ------------ 私有函数 -------------
    # 初始化目标图像
    def _initChImg(self):
        self.chImg = cv2.copyTo(self.img, self.chImg)

    # 生成噪点的位置
    def _GenRandomPoint(self):
        h, w = self.img.shape[:2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        return x, y

    # 高斯噪点
    def _GaussionNoiseAlgorithm(self, x, y):
        return self.chImg[y][x] + random.gauss(self.means, self.sigma)

    # 椒盐噪点
    def _SpicySaltAlgorithm(self, x, y):
        if (random.random() < 0.5):
            return 0
        return 255

    # 检查像素值是否合理，并修正
    def _CheckVal(self, val):
        if val < 0:
            return 0
        if val > 255:
            return 255
        return int(val)

    # 同一产生噪声
    def _GenNoise(self, func):
        for i in range(self.noiseCnt):
            x, y = self._GenRandomPoint()
            self.chImg[y][x] = self._CheckVal(func(x, y))

    # 输出图片
    def _ShowPic(self, img):
        print("Picture", img)
        print("Shape", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def _ShowMergePic(self, imgSrc, chImg):
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 使用高斯分布产生噪点(灰度)
    def CaussionNoise(self, means, sigma, rate):
        h, w = self.img.shape[:2]
        self.means = means
        self.sigma = sigma
        self.noiseCnt = int(h * w * rate)
        self._initChImg()
        self._GenNoise(self._GaussionNoiseAlgorithm)

    # 产生椒盐噪声
    def SpicySaltNoise(self, rate):
        h, w = self.img.shape[:2]
        self.noiseCnt = int(h * w * rate)
        self._initChImg()
        self._GenNoise(self._SpicySaltAlgorithm)

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.img)

    # 输出转换后的图像
    def ShowDstImg(self):
        self._ShowMergePic(self.img, self.chImg)

'''
以下为产生为噪声产生应用
'''
if __name__ == "__main__":
    myImg = MyImg('pythonProject1/pic/lenna.png')
    # 在灰度图基础上，生成高斯随机噪点
    myImg.CaussionNoise(2, 4, 0.8)
    myImg.ShowDstImg()

    # # 在灰度图基础上，生成椒盐随机噪点
    myImg.SpicySaltNoise(0.2)
    myImg.ShowDstImg()
