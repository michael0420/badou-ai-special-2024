'''
@author linrenkang
SIFT 算法
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MyImg:
    # 原图像
    img = None
    # 灰度图
    imgGray = None
    # 标注关键点的图像
    chImg = None

    # # SIFT 中间信息
    # keypoints = None

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # ------------ 私有函数 -------------
    # 输出图片
    def _ShowPic(self, img):
        print("Picture", img)
        print("Shape", img)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 图像拼接
    def _MatchImg(self, img1, kp1, des1, img2, kp2, des2):
        # 创建 K 临近算法（欧式距离）
        bf = cv2.BFMatcher(cv2.NORM_L2)
        # 找出相似度最高的前 k 个
        matchs = bf.knnMatch(des1, des2, k=2)

        # 筛选只保留一个距离
        selectMatchs = []
        for m, n in matchs:
            if m.distance < 0.5 * n.distance:
                selectMatchs.append([m])
        # 两个图像，与特征匹配标注到图上
        return cv2.drawMatchesKnn(img1, kp1, img2, kp2, selectMatchs, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # SIFT 核心算法调用
    def _SIFT(self, img):
        # 初始化 SIFT 检测器
        sift = cv2.xfeatures2d_SIFT.create()
        # SIFT 检测
        keypoints, descriptors = sift.detectAndCompute(img, None)
        # 将关键点标注到图像上
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 对图像的每个关键点都绘制了圆圈和方向
        chImg = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoints, descriptors, chImg

    #----------------- 导出接口 -------------
    # 在灰度图上，进行SIF
    def siftAlgorithmGray(self):
        _, _, self.chImg = self._SIFT(self.imgGray)

    # zai RGB图上，进行SIFT
    def siftAlgorithmRGB(self):
        _, _, self.chImg = self._SIFT(self.img)

    # 图像特征匹配
    def myMatch(self, path1, path2):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        keypoints1, descriptors1, _ = self._SIFT(img1)
        keypoints2, descriptors2, _ = self._SIFT(img2)
        self.chImg = self._MatchImg(img1, keypoints1, descriptors1, img2, keypoints2, descriptors2)

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.img)

    # 输出目标图像
    def ShowDstImg(self):
        self._ShowPic(self.chImg)

'''
以下 SIFT 的应用
'''
if __name__ == "__main__":
    myImg = MyImg('pic/lenna.png')
    # 输出原图
    myImg.ShowSrcImg()

    # 灰度图计算 SIFT
    myImg.siftAlgorithmGray()
    myImg.ShowDstImg()

    # RGB 计算 SIFT
    myImg.siftAlgorithmRGB()
    myImg.ShowDstImg()

    # 图像拼接
    myImg.myMatch('pic/iphone1.png', 'pic/iphone2.png')
    myImg.ShowDstImg()
