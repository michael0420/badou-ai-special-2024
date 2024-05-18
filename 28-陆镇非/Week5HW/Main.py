# Author: Zhenfei Lu
# Created Date: 4/26/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import time
from Utils import ImageUtils
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(999999999)

class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test8(self):
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        greyImage = ImageUtils.BGRImage2GreyImage(BGRImage)
        img_canny = ImageUtils.canny(img=greyImage, sigma=0.5, guassianFilterDim=5, low_boundary=60, high_boundary=180)
        dict2 = {}
        dict2['canny'] = img_canny
        ImageUtils.plotAllRGBImages(dict2, False)
        return
        # openCV existed libs:
        # kernel_size = 5
        # lowThres, highThres = (200, 300)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.Canny(gray, lowThres, highThress, apertureSize = kernel_size)

    def test9(self):
        imageFilePath = "./photo1.jpg"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        xyIndexArr = ImageUtils.connerPointDetect(BGRImage, 4)
        ijIndexArr = ImageUtils.xyPlane2RowColPlane(xyIndexArr)
        desiredImgSize = tuple((800, 600, 3))
        originalPixelIndexList = ImageUtils.getClockWise4PixelIndex(ijIndexArr)
        desiredPixelIndexList = ImageUtils.generateClockWise4PixelIndexByImgSize(desiredImgSize)
        # originalPixelIndexList = np.array([[151, 207], [285, 517], [601, 17], [731, 343]])   # i-j plane , img[i,j]
        # desiredPixelIndexList = np.array([[0, 0], [0, 337], [488, 0], [488, 337]])
        # desiredImgSize = tuple((488, 337, 3))
        # originalPixelIndexList = np.array([[207, 151], [517, 285], [17, 601], [343, 731]])  # x-y plane. img[y,x]
        # desiredPixelIndexList = np.array([[0, 0], [337, 0], [0, 488], [337, 488]])
        # desiredImgSize = tuple((337, 488, 3))
        mat = ImageUtils.getAffineMatrix(originalPixelIndexList, desiredPixelIndexList)
        desiredImg = ImageUtils.affineTransfer(BGRImage, mat, desiredImgSize)
        print(mat)
        dict2 = {}
        dict2['original'] = ImageUtils.BGRImg2RGBImg(BGRImage)
        dict2['affine transfer'] = ImageUtils.BGRImg2RGBImg(desiredImg)
        ImageUtils.plotAllRGBImages(dict2, False)
        return
        # openCV existed libs:
        # m = cv2.getPerspectiveTransform(src, dst)
        # result = cv2.warpPerspective(result3, m, (337, 488))

    def runAlltests(self) -> None:
        # test8
        start_time = time.time()
        self.test8()
        end_time = time.time()
        print("test8 excuted time cost：", end_time - start_time, "seconds")

        # test9
        start_time = time.time()
        self.test9()
        end_time = time.time()
        print("test9 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")


if __name__ == "__main__":
    solution = Solution()
