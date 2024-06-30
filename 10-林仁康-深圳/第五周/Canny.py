'''
@author linrenkang
Canny 边缘提取
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 灰度图
    grayImg = None
    # 降噪图像
    reduceImg = None
    # 边缘图像
    edgeImg = None
    # 非极大值抑制后的图像
    selectImg = None
    # 缩放后的图像
    chImg = None

    # 梯度斜率(float)
    tan = None
    # 孤立弱边缘消除栈
    sp = []
    # 双阈值
    minThrehold = 0
    maxThrehold = 0
    # 高斯卷积核大小
    GAUSSIAN_MATRIX = (5, 5)

    def __init__(self, path):
        self.img = cv2.imread(path)
        cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR, cv2.CV_32FC3)

    # ------------ 私有函数 -------------
    # 初始化
    def _Init(self):
        self.sp.clear()
    # 设置阈值
    def _SetDoubleThrehold(self, minThrehold, maxThrehold):
        self.minThrehold = minThrehold
        self.maxThrehold = maxThrehold

    # 求斜率
    def _SetTan(self, x, y):
        self.tan = y / x
        self.tan[self.tan == 0] = 0.000001

    # 获取斜率与周围交点值
    def _GetPixelTan(self, x, y):
        # 周围点
        # adjPoint = self.self.edgeImg[y - 1:y + 2, x - 1:x + 2]

        h, w = self.selectImg.shape[:2]
        # 斜率
        k = self.tan[y, x]

        # 坐标
        left = x - 1 if x - 1 >= 0 else 0
        right = x + 1 if x + 1 < w else x
        up = y - 1 if y - 1 >= 0 else 0
        down = y + 1 if y + 1 < h else y

        # 线性插值
        if k <= -1:
            # 二、四象限，y >= x(使用 x 轴上的等比变换，计算像素)
            pixel0 = self.selectImg[up, x] + (self.selectImg[up, x] - self.selectImg[up, left]) / k
            pixel1 = self.selectImg[down, x] + (self.selectImg[down, x] - self.selectImg[down, right]) / k
            return pixel0, pixel1
        if k >= 1:
            # 一、三象限，y >= x(使用 x 轴上的等比变换，计算像素)
            pixel0 = self.selectImg[up, x] + (self.selectImg[up, right] - self.selectImg[up, x]) / k
            pixel1 = self.selectImg[down, x] + (self.selectImg[down, left] - self.selectImg[down, x]) / k
            return pixel0, pixel1
        if k < 0:
            # 二、四象限，y < x(使用 y 轴上的等比变换，计算像素)
            pixel0 = self.selectImg[y, left] + (self.selectImg[y, left] - self.selectImg[up, left]) / k
            pixel1 = self.selectImg[y, right] + (self.selectImg[y, right] - self.selectImg[down, right]) / k
            return pixel0, pixel1
        # 一、三象限，y < x(使用 y 轴上的等比变换，计算像素)
        pixel0 = self.selectImg[y, left] + (self.selectImg[down, left] - self.selectImg[y, left]) / k
        pixel1 = self.selectImg[y, right] + (self.selectImg[up, right] - self.selectImg[y, right]) / k
        return pixel0, pixel1

    # 判断是否为弱边缘
    def _IsStrongEdge(self, val):
        return val >= self.maxThrehold

    # 判断是否为强边缘
    def _IsWeakEdge(self, val):
        return self.minThrehold < val < self.maxThrehold

    # 检查是否非边缘值
    def _IsNotEdge(self, val):
        return val <= self.minThrehold

    def _ClearWeakEage(self):
        h, w = self.selectImg.shape[:2]
        for y in range(h):
            for x in range(w):
                if self._IsWeakEdge(self.selectImg[y, x]):
                    self.selectImg[y, x] = 0

    # 双阈值检测
    def _DoubleThreholdDetect(self, x, y):
        val = self.selectImg[y, x]
        if self._IsStrongEdge(val):
            self.sp.append((x, y))
            return 255
        if self._IsNotEdge(val):
            return 0
        return val

    # 设置孤立边缘
    def _SetIsolationEdage(self, x, y):
        val = self.selectImg[y, x]
        if self._IsWeakEdge(val):
            self.sp.append((x, y))
            return 255
        return val

    # 检查是否为孤立的边缘点
    def _IsIsolationEdge(self, x, y):
        pixelAdj = self.selectImg[y - 1:y + 2, x - 1:x + 2]
        for row in pixelAdj:
            for element in row:
                if self._IsStrongEdge(element):
                    return True
        return False

    # 灰度化
    def _ImgToGray(self):
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.ShowPicName(self.grayImg, "Gray")

    # 降噪：高斯降噪
    def _ReduceNoise(self):
        self.reduceImg = cv2.GaussianBlur(self.grayImg, self.GAUSSIAN_MATRIX, 0)
        # self.ShowPicName(self.reduceImg, "Gaussian")

    # 边缘检测:sobel
    def _AllEdgeDetection(self):
        # x 方向的 sobel
        xTmp = cv2.Sobel(self.reduceImg, cv2.CV_16S, 1, 0)
        xImg = cv2.convertScaleAbs(xTmp)
        # self.ShowPicName(xImg, "Sobel X Detect")
        # y 方向上的 sobel
        yTmp = cv2.Sobel(self.reduceImg, cv2.CV_16S, 0, 1)
        yImg = cv2.convertScaleAbs(yTmp)
        # self.ShowPicName(yImg, "Sobel Y Detect")
        # 合并 X 和 Y 方向上的 sobel 边缘检测图像
        self.edgeImg = cv2.addWeighted(xImg, 0.5, yImg, 0.5, 0)
        self.ShowPicName(self.edgeImg, "Sobel Detect")

        # 同时设置梯度斜率
        xImg = xImg.astype(np.float32)
        yImg = yImg.astype(np.float32)
        xImg[xImg == 0] = 0.00000001
        self._SetTan(xImg, yImg)

    # 设置 sobel 矩阵
    def _GetSobel(self, img, kenel):
        return np.sum(img * kenel)

    def _AllEdgeDetectionDetail(self):
        h, w = self.reduceImg.shape[:2]
        # 初始化
        xSobel = np.zeros(self.reduceImg.shape)
        ySobel = np.zeros(self.reduceImg.shape)
        self.edgeImg = np.zeros(self.reduceImg.shape)
        self.reduceImg = np.pad(self.reduceImg, ((1, 1), (1, 1)), 'constant')

        # sobel 矩阵
        sobelXKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelYKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        for i in range(h):
            for j in range(w):
                xSobel[i, j] = self._GetSobel(self.reduceImg[i:i + 3, j:j + 3], sobelXKernel)
                ySobel[i, j] = self._GetSobel(self.reduceImg[i:i + 3, j:j + 3], sobelYKernel)
                self.edgeImg[i, j] = np.sqrt(xSobel[i, j] ** 2 + ySobel[i, j] ** 2)
        self.ShowPicName(self.edgeImg.astype(np.uint8), "Sobel Detect")

        # 同时设置梯度斜率
        xImg = xSobel.astype(np.float32)
        yImg = ySobel.astype(np.float32)
        xImg[xImg == 0] = 0.00000001
        self._SetTan(xImg, yImg)

    # 非极大值抑制
    def _RemoveEdge(self):
        h, w = self.edgeImg.shape[:2]
        self.selectImg = self.edgeImg.copy().astype(np.float32)
        for y in range(h - 1):
            for x in range(w - 1):
                pixel0, pixel1 = self._GetPixelTan(x, y)
                # 抑制极大值
                if not (self.selectImg[y, x] > pixel0 and self.selectImg[y, x] > pixel1):
                    self.selectImg[y, x] = 0
        self.ShowPicName(self.selectImg.astype(np.uint8), "Non maximum suppression")

    # 边缘检测：双阈值检测
    def _SelectEdge(self):
        h, w = self.selectImg.shape[:2]
        for y in range(h - 1):
            for x in range(w - 1):
                self.selectImg[y, x] = self._DoubleThreholdDetect(x, y)
        self.ShowPicName(self.selectImg.astype(np.uint8), "Double threshold detection")

    # 抑制孤立的弱边缘
    def _RemoveIsolationEdge(self):
        while len(self.sp) != 0:
            x, y = self.sp.pop()
            # 周围的点
            yList = [y - 1, y, y + 1]
            xList = [x - 1, x, x + 1]
            for i in enumerate(yList):
                for j in enumerate(xList):
                    if i == y and j == x:
                        continue
                    self.selectImg[y, x] = self._SetIsolationEdage(x, y)
        self._ClearWeakEage()
        self.ShowPicName(self.selectImg.astype(np.uint8), "Suppress isolated weak edges")

    # 输出图片
    def ShowPic(self, img):
        print("Picture: ", img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出图片
    def ShowPicName(self, img, name):
        print("Picture: " + name, img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture: " + name, img)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def ShowMergePic(self, imgSrc, chImg):
        self.ShowPic(chImg)
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 根据目标规模缩放
    def Canny(self):
        self._Init()
        # 1 设置灰度图
        self._ImgToGray()
        # 2 高斯降噪
        self._ReduceNoise()
        # 3 sobel 边缘检测
        # self._AllEdgeDetection()
        self._AllEdgeDetectionDetail()
        # 4 极大值抑制
        self._RemoveEdge()
        # 5 双阈值检测
        # 设置阈值
        minThrehold = self.edgeImg.mean() * 0.4
        maxThrehold = minThrehold * 3
        print("minThrehold: ", minThrehold)
        print("maxThrehold: ", maxThrehold)
        self._SetDoubleThrehold(minThrehold, maxThrehold)
        self._SelectEdge()
        # 6 抑制孤立弱边缘
        self._RemoveIsolationEdge()

    # 输出原图
    def ShowSrcImg(self):
        self.ShowPic(self.img)

    # 输出转换后的图像（彩色图）
    def ShowDstImg(self):
        self.ShowMergePic(self.img, self.chImg)

'''
以下为 Canny 边缘提取的应用
'''

if __name__ == "__main__":
    myImg = MyImg('pic/lenna.png')
    # 输出原图
    myImg.ShowSrcImg()

    # canny 检测
    myImg.Canny()
