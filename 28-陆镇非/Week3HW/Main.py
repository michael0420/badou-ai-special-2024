# Author: Zhenfei Lu
# Created Date: 4/1/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import time
from Utils import ImageUtils

class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test3(self) -> None:
        # grey image 1 channel histogram test
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        greyImage = ImageUtils.BGRImage2GreyImage(BGRImage)
        (FilteredImage, histogramDictOriginal) = ImageUtils.histogramEqualFilterC1(greyImage)
        histogramDictFiltered = ImageUtils.getHistogramFromImageC1(FilteredImage)
        histogramDictDict = dict()
        histogramDictDict["histogram original"] = histogramDictOriginal
        histogramDictDict["histogram filtered"] = histogramDictFiltered
        ImageUtils.plotAllHistograms(histogramDictDict, False)
        dict1 = {}
        dict1['grey image '] = greyImage
        dict1['histogramed grey image '] = FilteredImage
        ImageUtils.plotAllRGBImages(dict1, False)
        # 3 channels rgb histogram test
        RGBImage = ImageUtils.BGRImg2RGBImg(BGRImage)
        histogramDictOriginal_RGB = list()
        histogramDictFiltered_RGB = list()
        FilteredRGBImage = np.zeros(RGBImage.shape, RGBImage.dtype)
        for i in range(0, 3):
            (FilteredImage, histogramDictOriginal) = ImageUtils.histogramEqualFilterC1(RGBImage[:, :, i])
            histogramDictFiltered = ImageUtils.getHistogramFromImageC1(FilteredImage)
            FilteredRGBImage[:, :, i] = FilteredImage
            histogramDictOriginal_RGB.append(histogramDictOriginal)
            histogramDictFiltered_RGB.append(histogramDictFiltered)
        dict2 = {}
        dict2['original RGB image '] = RGBImage
        dict2['histogramed RGB image '] = FilteredRGBImage
        ImageUtils.plotAllRGBImages(dict2, False)
        RBG_arr = ["R","G","B"]
        for i in range(0, 3):
            histogramDictDict = dict()
            histogramDictDict["histogram-original channel " + RBG_arr[i]] = histogramDictOriginal_RGB[i]
            histogramDictDict["histogram-filtered channel " + RBG_arr[i]] = histogramDictFiltered_RGB[i]
            ImageUtils.plotAllHistograms(histogramDictDict, False)
        return
        # openCV existed libs:
        # dst = cv2.equalizeHist(gray)
        # hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    def test4(self) -> None:
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        greyImage = ImageUtils.BGRImage2GreyImage(BGRImage)
        gd_x_img = ImageUtils.convolutionFilterC1(greyImage, ImageUtils.sobelX, stride=1, padding=1)
        gd_y_img = ImageUtils.convolutionFilterC1(greyImage, ImageUtils.sobelY, stride=1, padding=1)
        # method1: add float first, then normalize.
        # sobel_img_float = ImageUtils.imageWeightedAdd(((gd_x_img, 0.5), (gd_y_img, 0.5)))
        # sobel_img = ImageUtils.normalizeImageC1(sobel_img_float)
        # method2: normalize first, then add , finally normalize again.  method2 is better in output image
        normalized_x_img = ImageUtils.normalizeImageC1(gd_x_img)
        normalized_y_img = ImageUtils.normalizeImageC1(gd_y_img)
        sobel_img_float = ImageUtils.imageWeightedAdd(((normalized_x_img, 0.5), (normalized_y_img, 0.5)))
        sobel_img = ImageUtils.normalizeImageC1(sobel_img_float)
        dict2 = {}
        dict2['grey image '] = greyImage
        dict2['sobel image '] = sobel_img
        ImageUtils.plotAllRGBImages(dict2, False)
        return
        # openCV existed libs:
        # cv2.Sobel(img, cv2.CV_16S, 1, 0)
        # cv2.convertScaleAbs(x)
        # cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def test5(self) -> None:
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        RGBImage = ImageUtils.BGRImg2RGBImg(BGRImage)
        targetSize = (361, 220)
        interporatedImg = ImageUtils.biLinearInterpolation(RGBImage, targetSize)
        (original_h, original_w) = RGBImage.shape[0:2]
        dict2 = {}
        dict2['original RGB image ' + str(original_h) + 'x' + str(original_w)] = RGBImage
        dict2['bi-interp image ' + str(targetSize[0]) + 'x' + str(targetSize[1])] = interporatedImg
        ImageUtils.plotAllRGBImages(dict2, False)

    def runAlltests(self) -> None:
        # test3
        start_time = time.time()
        self.test3()
        end_time = time.time()
        print("test3 excuted time cost：", end_time - start_time, "seconds")

        # test4
        start_time = time.time()
        self.test4()
        end_time = time.time()
        print("test4 excuted time cost：", end_time - start_time, "seconds")

        # test5
        start_time = time.time()
        self.test5()
        end_time = time.time()
        print("test5 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")

if __name__ == "__main__":
    solution = Solution()
