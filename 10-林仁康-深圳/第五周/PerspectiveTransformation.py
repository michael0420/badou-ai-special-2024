'''
@author linrenkang
透视变换
'''

import cv2
import numpy as np


class MyImg:
    # 原图像
    img = None
    # 变换图像
    chImg = None
    dstW = 0
    dstH = 0

    # 计算矩阵
    A = None
    B = None
    T = None

    # 输入的个数
    POINT_CNT = 4
    A_ROW = 8
    A_COL = 8
    B_ROW = 8
    B_COL = 1

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.A = np.zeros((self.A_ROW, self.A_COL))
        self.B = np.zeros((self.B_ROW, self.B_COL))

    # ------------ 私有函数 -------------
    # 根据输入的坐标，计算 A 和 B 矩阵
    def _SetAB(self, src, dst):
        assert src.shape[0] >= self.POINT_CNT and dst.shape[0] >= self.POINT_CNT
        for i in range(self.POINT_CNT):
            # 原坐标与目的坐标
            srcX, srcY = src[i, 0], src[i, 1]
            dstX, dstY = dst[i, 0], dst[i, 1]
            # 设置参数
            self.A[i * 2] = [srcX, srcY, 1, 0, 0, 0, -srcX * dstX, -srcY * dstX]
            self.A[i * 2 + 1] = [0, 0, 0, srcX, srcY, 1, -srcX * dstY, -srcY * dstY]
            self.B[i * 2] = dstX
            self.B[i * 2 + 1] = dstY
        self.A = np.mat(self.A)
        self.B = np.mat(self.B)

    # 根据 A 和 B 矩阵，计算 T
    def _SetT(self):
        self.T = np.array(self.A.I * self.B).T[0]
        self.T = np.insert(self.T, self.T.shape[0], values=1, axis=0)
        self.T = np.mat(self.T.reshape((3, 3)))

    # 设置目标图像大小
    def _SetSize(self, h, w):
        self.dstH = h
        self.dstW = w
        self.chImg = np.zeros((h, w, self.img.shape[2]))

    # 获取目标坐标
    def _GetDstPoint(self, x, y):
        inputM = np.mat([x, y, 1])[0].T
        outputM = np.array(self.T * inputM).T[0]
        xNew, yNew, zNew = outputM[0], outputM[1], outputM[2]
        return round(xNew / zNew), round(yNew / zNew)

    # 获取源坐标
    def _GetSrcPoint(self, x, y):
        inputM = np.mat([x, y, 1])[0].T
        outputM = np.array(self.T.I * inputM).T[0]
        xNew, yNew, zNew = outputM[0], outputM[1], outputM[2]
        return round(xNew / zNew), round(yNew / zNew)

    # 越界处理
    def _ExpectHandle(self, x, y, h=0, w=0):
        if h == 0:
            h = self.dstH
        if w == 0:
            w = self.dstW

        x = x if x > 0 else 0
        y = y if y > 0 else 0

        x = x if x < w else w - 1
        y = y if y < h else h - 1

        return x, y

    # 识别越界处理
    def _IsExpectVal(self, x, y, h=0, w=0):
        if h == 0:
            h = self.dstH
        if w == 0:
            w = self.dstW

        if x < 0 or x >= w:
            return True
        if y < 0 or y >= h:
            return True

        return False

    # 验算目的坐标
    def _DstCheck(self, src, dst):
        print("src: ", src)
        dstTmp = np.zeros(src.shape)
        for i in range(src.shape[0]):
            x, y = src[i, 0], src[i, 1]
            xNew, yNew = self._GetDstPoint(x, y)
            dstTmp[i, 0], dstTmp[i, 1] = xNew, yNew
        print("Check dstTmp: ", dstTmp)
        print("Check dst: ", dst)

    # 验算源坐标
    def _SrcCheck(self, src, dst):
        print("dst: ", dst)
        srcTmp = np.zeros(src.shape)
        for i in range(dst.shape[0]):
            x, y = dst[i, 0], dst[i, 1]
            xNew, yNew = self._GetSrcPoint(x, y)
            srcTmp[i, 0], srcTmp[i, 1] = xNew, yNew
        print("Check srcTmp: ", srcTmp)
        print("Check src: ", src)

    # 输出图片
    def _ShowPic(self, img):
        print("Picture: ", img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture", img.astype(np.uint8))
        cv2.waitKey(0)

    # 输出图片
    def _ShowPicName(self, img, name):
        print("Picture: " + name, img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture: " + name, img.astype(np.uint8))
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def _ShowMergePic(self, imgSrc, chImg):
        self._ShowPic(chImg)
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 训练透视变换函数
    def PerTransformDetail(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        self._SetAB(src, dst)
        self._SetT()
        self._DstCheck(src, dst)
        self._SrcCheck(src, dst)
        print("PerTransformDetail T: ", self.T)

    # 根据训练的 T 进行透视变换(正向计算)
    def WarpPerspectiveDetailForward(self, dstH, dstW):
        h, w = self.img.shape[:2]
        self._SetSize(dstH, dstW)
        self.chImg = np.zeros((dstH, dstW, self.img.shape[2]))
        # 累计计算为同一点坐标
        pointCnt = np.zeros((dstH, dstW))

        # 对原图像计算
        for y in range(h):
            for x in range(w):
                xNew, yNew = self._GetDstPoint(x, y)
                # 丢弃越界的点
                if self._IsExpectVal(xNew, yNew):
                    continue
                # 处理越界的点
                # xNew, yNew = self._ExpectHandle(xNew, yNew)
                # 移动坐标
                self.chImg[yNew, xNew] = self.chImg[yNew, xNew] + self.img[y, x]
                pointCnt[yNew, xNew] += 1

        # 多个对应点取平均值
        for y in range(dstH):
            for x in range(dstW):
                if pointCnt[y, x] > 1:
                    self.chImg[y, x] = self.chImg[y, x] / pointCnt[y, x]

    # 根据训练的 T 进行透视变换（逆向计算）
    def WarpPerspectiveDetailReverse(self, dstH, dstW):
        h, w = self.img.shape[:2]
        self._SetSize(dstH, dstW)
        self.chImg = np.zeros((dstH, dstW, self.img.shape[2]))

        # 对原图像计算
        for yNew in range(dstH):
            for xNew in range(dstW):
                x, y = self._GetSrcPoint(xNew, yNew)
                # 丢弃越界的点
                if self._IsExpectVal(x, y, h, w):
                    continue
                # 移动坐标
                # print("x, y, xNew, yNew", x, y, xNew, yNew)
                self.chImg[yNew, xNew] = self.img[y, x]


    # 训练透视变换函数
    def PerTransform(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        self.T = cv2.getPerspectiveTransform(src, dst)
        print("PerTransform T: ", self.T)

    def WarpPerspective(self, dstH, dstW):
        self.chImg = cv2.warpPerspective(self.img, self.T, (dstH, dstW))

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.img)

    # 输出转换后的图像（彩色图）
    def ShowDstImg(self):
        self._ShowPic(self.chImg)

'''
以下为透视变换的应用
'''

if __name__ == "__main__":
    myImg = MyImg('pic/photo1.jpg')
    # 输出原图
    myImg.ShowSrcImg()
    print("img: [207, 151]", myImg.img[207, 151])

    # 设置源目地址
    # src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    # dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]

    src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    dst = [[0, 0], [337, 0], [0, 488], [337, 488]]
    myImg.PerTransformDetail(src, dst)

    # 正向计算透视变换
    myImg.WarpPerspectiveDetailForward(500, 500)
    myImg.ShowDstImg()

    # 逆向计算透视变换
    myImg.WarpPerspectiveDetailReverse(500, 500)
    myImg.ShowDstImg()

    # 使用 cv2 进行透视变换
    myImg.PerTransform(src, dst)
    myImg.WarpPerspective(500, 500)
    myImg.ShowDstImg()
