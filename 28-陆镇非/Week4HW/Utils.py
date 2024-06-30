# Author: Zhenfei Lu
# Created Date: 4/1/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageUtils(object):
    @staticmethod
    def readImgFile2BGRImage(filePath: str) -> np.ndarray:
        img = cv2.imread(filePath)  # openCV existed lib
        return img

    @staticmethod
    def BGRImg2RGBImg(img: np.ndarray) -> np.ndarray:
        (h, w, c) = img.shape
        RGBImg = np.zeros((h, w, c), img.dtype)
        RGBImg[:, :, 0] = img[:, :, 2]
        RGBImg[:, :, 1] = img[:, :, 1]
        RGBImg[:, :, 2] = img[:, :, 0]
        return RGBImg
    # RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)  # openCV existed lib

    @staticmethod
    def plotAllRGBImages(imgDict: dict, showImmediately:bool = True) -> None:
        N = len(imgDict)
        plt.figure()
        i = 1
        for key, value in imgDict.items():
            plt.subplot(1, N, i)
            plt.imshow(value, cmap='gray') # matplot lib only accepts RGB order image
            plt.title(key)
            plt.xticks([])
            plt.yticks([])
            i = i + 1
        if(showImmediately):
            plt.show()

    @staticmethod
    def showAllPlotsImmediately(showImmediately: bool = False) -> None:
        if (showImmediately):
            plt.show()

    @staticmethod
    def BGRImage2GreyImage(img: np.ndarray) -> np.ndarray:
        (h, w) = img.shape[0:2]
        greyImg = np.zeros((h, w), img.dtype)
        for i in range(0, h):
            for j in range(0, w):
                greyImg[i, j] = int(img[i, j, 0]*0.11 + img[i, j, 1]*0.59 + img[i, j, 2]*0.3)  # BGR order
        return greyImg
        # # openCV existed lib
        # greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return greyImg


    @staticmethod
    def greyImage2BinaryImage(greyImage: np.ndarray) -> np.ndarray:
        (h, w) = greyImage.shape
        BinaryImage = np.zeros((h, w), dtype=np.float)
        for i in range(0, h):
            for j in range(0, w):
                val = greyImage[i, j] / 255
                if(val > 0.5):
                    BinaryImage[i, j] = 1
                else:
                    BinaryImage[i, j] = 0
        return BinaryImage
        # return np.where(img_gray >= 0.5, 1, 0) # more simple way to use numpy.where

    @staticmethod
    def BGRImageNearestInteropate(img: np.ndarray, targetImgSize: tuple) -> np.ndarray:
        (h, w, channels) = img.shape[0:3]
        (h_target, w_target) = targetImgSize[0:2]
        h_stepSize = h / h_target
        w_stepSize = w / w_target
        targetImage = np.zeros((h_target, w_target, channels), img.dtype)
        for i in range(0, h_target):
            for j in range(0, w_target):
                original_pixel_index_i = int(h_stepSize * i + 0.5)
                original_pixel_index_j = int(w_stepSize * j + 0.5)
                if(original_pixel_index_i > h - 1):  # more robust for index exceeding the max index
                    original_pixel_index_i = h - 1
                if (original_pixel_index_j > w - 1):
                    original_pixel_index_j = w - 1
                targetImage[i, j, :] = img[original_pixel_index_i, original_pixel_index_j, :]
        return targetImage

    @staticmethod
    def histogramEqualFilterC1(img: np.ndarray) -> tuple:
        histogramDictOriginal = ImageUtils.getHistogramFromImageC1(img)
        (h, w) = img.shape[0:2]  # get size again
        outputImg = np.zeros((h, w), img.dtype)  # be same size
        histogramDesire_i = (h * w) / 256  # each desireValue(Number) of histogram should be histogram Equal. Equally assign the number(y-axis) for pixelValue(x-axis)
        # ∑histo_input = ∑histo_output = (i+1)*histogramDesire_i
        # i = ((∑histo_input) / histogramDesire_i) - 1
        histogram_sum = 0
        N = len(histogramDictOriginal) # must = 256
        for j in range(0, N):
            N_pixelValue = len(histogramDictOriginal[j])
            histogram_sum = histogram_sum + N_pixelValue
            i = (histogram_sum / histogramDesire_i) - 1  # i is outputImage's pixelValue(x-axis) of histogram, float type
            if(i > 255): # robust for value judge
                i = 255
            if(i < 0):
                i = 0
            for pixelIndex in histogramDictOriginal[j]:
                (row, column) = pixelIndex  # search the pixel index in the original image
                outputImg[row, column] = int(i)  # get int
        return tuple((outputImg, histogramDictOriginal))

    @staticmethod
    def getHistogramFromImageC1(img) -> dict:
        (h, w) = img.shape[0:2]
        histogramDict = dict()
        N = 256
        for i in range(0, N):
            histogramDict[i] = list(tuple())
        for i in range(0, h):
            for j in range(0, w):
                pixelVal = img[i, j]
                histogramDict[pixelVal].append(tuple((i, j)))
        return histogramDict

    @staticmethod
    def plotAllHistograms(histogramDictDict: dict, showImmediately:bool = True) -> None:
        plt.figure()
        i = 1
        N_total = len(histogramDictDict)
        for key, value in histogramDictDict.items():
            title = key
            histogramDict = value
            N = len(histogramDict) # must = 256, 0-255
            x_list = list()
            y_list = list()
            for j in range(0, N):
                x_list.append(j)
                N_pixelValue = len(histogramDict[j])
                y_list.append(N_pixelValue)
            plt.subplot(1, N_total, i)
            plt.plot(x_list, y_list)
            # plt.hist(np.array(y_list), N)
            plt.title(title)
            plt.xlabel('pixelValue')
            plt.ylabel('Number of pixelValue')
            i = i + 1
        if (showImmediately):
            plt.show()

    @staticmethod
    def convolutionFilterC1(img: np.ndarray, filterKernel: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray: # padding=1 is zero-padding
        (h, w) = img.shape
        (h_f, w_f) = filterKernel.shape
        outputImg = np.zeros((int(((h-h_f+2*padding)/stride)+1), int(((w-w_f+2*padding)/stride))+1), dtype=np.float)
        paddingImg = np.zeros((h+2*padding, w+2*padding), img.dtype)
        (h_pad, w_pad) =paddingImg.shape
        paddingImg[(0+padding):(h_pad-padding), (0+padding):(w_pad-padding)] = img  # slice index is deep copy(value copy), will not change the original matrix
        window = np.zeros((h_f, w_f), paddingImg.dtype)
        row = 0
        col = 0
        for i in range(0, h + 2 * padding - (h_f - 1), stride):  # at the end of arr, step back (sizeOfFilter-1) for not exceeding the max index of arr
            col = 0
            for j in range(0, w + 2 * padding - (w_f - 1), stride):
                window = paddingImg[i:(i + (h_f - 1) + 1), j:(j + (w_f - 1) + 1)] # window start at the left-top corner
                pixelValue = ImageUtils.convolutionDotProductSum(window, filterKernel)
                outputImg[row, col] = pixelValue
                col = col + 1
            row = row + 1
        return outputImg

    @staticmethod
    def convolutionDotProductSum(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        return np.sum(np.sum(matrix1*matrix2, axis=1), axis=0) # scalar = sumAll(A.*B)

    @staticmethod
    def normalizeImageC1(img: np.ndarray) -> np.ndarray:  # convert float or exceed255 or minus image to [0-255]
        (h, w) = img.shape
        outputImg = np.zeros((h, w), dtype=np.int)
        outputImg = img.astype(int)
        for i in range(0, h):
            for j in range(0, w):
                pixelVal = outputImg[i, j]
                if(pixelVal > 255):
                    outputImg[i, j] = 255
                if(pixelVal < 0):
                    outputImg[i, j] = 0
        return outputImg

    @staticmethod
    def imageWeightedAdd(imgTuple: tuple) -> np.ndarray:
        img0 = imgTuple[0][0] # ((img0,w0),(img1,w1),(img3,w3)......)
        (h, w) = img0.shape
        outputImg = np.zeros((h, w), np.float) # multipy weight, so res is float
        for it in imgTuple:
            outputImg = outputImg + it[0] * it[1]
        return outputImg

    @staticmethod
    def biLinearInterpolation(img: np.ndarray, targetSize: tuple) -> np.ndarray:
        (h, w, c) = img.shape
        outputImg = np.zeros((targetSize[0], targetSize[1], c), img.dtype)
        h_stepSize = h / targetSize[0]
        w_stepSize = w / targetSize[1]
        for i in range(0, targetSize[0]):
            for j in range(0, targetSize[1]):
                i_middle = (i + 0.5) * h_stepSize + 0.5  # i_middle pixel index is in the original image
                j_middle = (j + 0.5) * w_stepSize + 0.5
                i_previous = min(int(np.floor(i_middle)), h-1)  # same  with if ixxx > 255: do sth.  robustness for index exceeding
                i_next = min(i_previous + 1, h-1)
                j_previous = min(int(np.floor(j_middle)), w-1)
                j_next = min(j_previous + 1, w-1)
                # print(i_middle, i_previous, i_next)
                # print(j_middle, j_previous, j_next)
                # for 2-D planner coordinate:  y_mid = (x1-x_mid)*y0/(x1-x0) + (x_mid-x0)*y1/(x1-x0)
                # fixed i value, get j_middle 's value
                f_iprevious_jmid = (j_next-j_middle)*img[i_previous, j_previous, :] + (j_middle-j_previous)*img[i_previous, j_next, :]
                f_inext_jmid = (j_next-j_middle)*img[i_next, j_previous, :] + (j_middle-j_previous)*img[i_next, j_next, :]
                # fixed j value(use f calculated before), get i_middle 's value
                f_imid_jmid = (i_next-i_middle)*f_iprevious_jmid + (i_middle-i_previous)*f_inext_jmid
                outputImg[i, j, :] = f_imid_jmid.astype(int)
        return outputImg

    # static properties:
    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
