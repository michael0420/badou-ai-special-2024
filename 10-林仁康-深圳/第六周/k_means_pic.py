'''
@author linrenkang
KMeans 在图像中的应用
'''
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 灰度图
    grayImg = None
    # 修改后的图像（灰度）
    chGrayImg = None
    # 修改后的图像（RGB）
    chRGBImg = []

    # 一维灰度图像
    zipGrayImg = None
    # 一维RGB图像
    zipRGBImg = None
    # 尝试次数
    MAX_TRY = 10

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # 压缩为一维灰度图
        w, h = self.grayImg.shape[:2]
        self.zipGrayImg = self.grayImg.reshape((w * h, 1))
        # 压缩为一维RGB图（行列长度自行计算，固定三维）
        _,_, c = self.img.shape
        self.zipRGBImg = self.img.reshape((-1, c))

    # ------------ 私有函数 -------------
    # 设置目标图像
    def _GetChImg(self, labels, centers, srcImgShape):
        # 浮点转 8 bit 整型
        centers = np.uint8(centers)
        # 根据簇的分类，使用质心像素替换原有的像素
        pixArry = centers[labels.flatten()]
        # # 浮点转 8 bit 整型
        # pixArry = np.int8(pixArry)
        return pixArry.reshape((srcImgShape))

    '''
    在OpenCV中，Kmeans()函数原型如下所示：
    compactness, labels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
        函数参数：
            data：需要分类数据，最好是np.float32数据，每个特征一列
            K：聚类个数
            bestLabels：预设分类标签，输出的整数数组，用于存储每个样本的聚类标签索引
            criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
                其中，type有如下模式：
                 —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足 epsilon 停止。
                 —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过 max_iter 停止。
                 —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
            attempts：重复试验kmeans算法的次数，返回最好一次的结果
            flags：初始中心的选择，可选以下两种
                --cv2.KMEANS_PP_CENTERS：使用 “kmeas++ 算法”的中心初始化算法
                --cv2.KMEANS_RANDOM_CENTERS：每次随机算去初始化中心
        返回值：
            compactness：紧密度，返回每个点到相应质心距离的平方和
            labels：结果标记，每个成员被标记分组的序号，如 0, 1, 2, 3 等等
            centers：有聚类中心组成的数组
    '''
    def _kMeans(self, srcImg, k):
        data = np.float32(srcImg)
        K = k
        bestLabels = None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.MAX_TRY, 1.0)
        attempts = self.MAX_TRY
        flags = cv2.KMEANS_RANDOM_CENTERS
        return cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)

    # 输出子图像
    def _ShowSubPic(self, img):
        if len(img.shape) == 2:
            plt.imshow(img, 'gray')
        else:
            plt.imshow(img)

    # 输出图片
    def _ShowPic(self, img):
        imgTmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Picture: ", imgTmp)
        print("Shape: ", imgTmp.shape)
        cv2.imshow("Picture", imgTmp)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def _ShowMergePic(self, title, imgArr):
        # 设置画布大小
        # plt.figure(dpi=300)
        picRow = round(len(imgArr) / 2)
        picCol = round(len(imgArr) / picRow)
        for i in range(len(imgArr)):
            plt.subplot(picRow, picCol, i + 1)
            plt.title(title[i])
            plt.xticks([])
            plt.yticks([])
            self._ShowSubPic(imgArr[i])
        plt.show()

    #----------------- 导出接口 -------------
    # 清空目标图像
    def clearChImg(self):
        self.chGrayImg = None
        self.chRGBImg.clear()
    
    # KMeans 算法（灰度图）
    def kMeansGray(self, k):
        _, labels, centers = self._kMeans(self.zipGrayImg, k)
        self.chGrayImg = self._GetChImg(labels, centers, self.grayImg.shape)

    # KMeans 算法（RGB图）
    def kMeansRGB(self, k):
        _, labels, centers = self._kMeans(self.zipRGBImg, k)
        chImg = self._GetChImg(labels, centers, self.img.shape)
        self.chRGBImg.append(chImg)

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.img)

    # 输出转换后的图像（彩色图）
    def ShowDstImgGray(self):
        title = ["Source Picture", "Changed Picture"]
        imgArr = [self.grayImg, self.chGrayImg]
        self._ShowMergePic(title, imgArr)

    # 输出转换后的图像（RGB 图）
    def ShowDstImgRGB(self):
        title = ['Source Picture']
        for i in range(len(self.chRGBImg)):
            titleStr = "Changed Picture" + str(i)
            title.append(titleStr)
        imgArr = copy.deepcopy(self.chRGBImg)
        imgArr.insert(0, self.img)
        self._ShowMergePic(title, imgArr)

'''
以下为 Keans 算法在图像中应用的实现
'''
if __name__ == "__main__":
    myImg = MyImg('pic/lenna.png')
    # 输出原图
    myImg.ShowSrcImg()

    # KMeans 检测（灰度）
    myImg.kMeansGray(4)
    myImg.ShowDstImgGray()

    # KMeans 检测（RGB）
    myImg.kMeansRGB(2)
    myImg.kMeansRGB(4)
    myImg.kMeansRGB(8)
    myImg.kMeansRGB(16)
    myImg.kMeansRGB(32)
    myImg.kMeansRGB(64)
    myImg.kMeansRGB(128)
    myImg.ShowDstImgRGB()
