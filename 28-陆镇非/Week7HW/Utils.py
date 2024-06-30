# Author: Zhenfei Lu
# Created Date: 4/27/2024
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
    def nearestInteropate(img: np.ndarray, targetImgSize: tuple) -> np.ndarray:
        (h, w) = img.shape[0:2]
        (h_target, w_target) = targetImgSize[0:2]
        h_stepSize = h / h_target
        w_stepSize = w / w_target
        targetImage = np.zeros(targetImgSize, img.dtype)
        for i in range(0, h_target):
            for j in range(0, w_target):
                original_pixel_index_i = int(h_stepSize * i + 0.5)
                original_pixel_index_j = int(w_stepSize * j + 0.5)
                if(original_pixel_index_i > h - 1):  # more robust for index exceeding the max index
                    original_pixel_index_i = h - 1
                if (original_pixel_index_j > w - 1):
                    original_pixel_index_j = w - 1
                targetImage[i, j] = img[original_pixel_index_i, original_pixel_index_j]   # img[i,j,:] is equal to this form img[i,j], different from matlab
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
    def getPaddingByKeepingSameDim(filterDim) -> int:
        return int((filterDim-1) / 2)

    @staticmethod
    def convolutionFilterC1(img: np.ndarray, filterKernel: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray: # padding=1 is zero-padding for 3x3 kernel case
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
        (h, w) = img.shape[0:2]
        outputImg = np.zeros(targetSize, img.dtype)
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
                f_iprevious_jmid = (j_next-j_middle)*img[i_previous, j_previous] + (j_middle-j_previous)*img[i_previous, j_next]
                f_inext_jmid = (j_next-j_middle)*img[i_next, j_previous] + (j_middle-j_previous)*img[i_next, j_next]
                # fixed j value(use f calculated before), get i_middle 's value
                f_imid_jmid = (i_next-i_middle)*f_iprevious_jmid + (i_middle-i_previous)*f_inext_jmid
                outputImg[i, j] = f_imid_jmid.astype(int)  # img[i,j,:] is equal to this form img[i,j], different from matlab
        return outputImg

    # static properties:
    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    @staticmethod
    def generateGaussianFilter(sigma, guassianFilterDim) -> np.ndarray:
        guassianFilter = np.zeros((guassianFilterDim, guassianFilterDim))
        oneDimArr = [i - guassianFilterDim // 2 for i in range(guassianFilterDim)]
        for i in range(0, guassianFilterDim):
            for j in range(0, guassianFilterDim):
                guassianFilter[i, j] = (1/(2*np.pi*sigma**2)) * np.exp((-1/(2*sigma**2)) * (oneDimArr[i]**2 + oneDimArr[j]**2))
        return guassianFilter/guassianFilter.sum()

    @staticmethod
    def convolutionFilterC1Multible(img: np.ndarray, filterKernel: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray:
        if(filterKernel.ndim < 3):
            return ImageUtils.convolutionFilterC1(img, filterKernel, stride, padding)
        (h, w) = img.shape
        (h_f, w_f) = filterKernel.shape[0:2]
        outputImg_num = filterKernel.shape[2]
        outputImg = np.zeros((int(((h-h_f+2*padding)/stride)+1), int(((w-w_f+2*padding)/stride))+1, outputImg_num), dtype=np.float)
        paddingImg = np.zeros((h+2*padding, w+2*padding), img.dtype)
        (h_pad, w_pad) = paddingImg.shape
        paddingImg[(0+padding):(h_pad-padding), (0+padding):(w_pad-padding)] = img  # slice index is deep copy(value copy), will not change the original matrix
        window = np.zeros((h_f, w_f), paddingImg.dtype)
        row = 0
        col = 0
        for i in range(0, h + 2 * padding - (h_f - 1), stride):  # at the end of arr, step back (sizeOfFilter-1) for not exceeding the max index of arr
            col = 0
            for j in range(0, w + 2 * padding - (w_f - 1), stride):
                for k in range(0, outputImg_num):  # saving time by doing convolution together when traverse the image
                    filterKernel_k = filterKernel[:, :, k]
                    window = paddingImg[i:(i + (h_f - 1) + 1), j:(j + (w_f - 1) + 1)] # window start at the left-top corner
                    pixelValue = ImageUtils.convolutionDotProductSum(window, filterKernel_k)
                    outputImg[row, col, k] = pixelValue
                col = col + 1
            row = row + 1
        return outputImg

    @staticmethod
    def canny(img: np.ndarray, sigma, guassianFilterDim, low_boundary, high_boundary) -> np.ndarray:
        (h, w) = img.shape
        # filter by guassian
        guassianFilter = ImageUtils.generateGaussianFilter(sigma, guassianFilterDim)
        img_g = ImageUtils.convolutionFilterC1Multible(img, guassianFilter, stride=1, padding=ImageUtils.getPaddingByKeepingSameDim(guassianFilterDim))

        # filter by sobel, and get gradient, tangent
        sobelFilter = np.zeros((3, 3, 2))
        sobelFilter[:, :, 0] = ImageUtils.sobelX
        sobelFilter[:, :, 1] = ImageUtils.sobelY
        img_sobel = ImageUtils.convolutionFilterC1Multible(img_g, sobelFilter, stride=1, padding=1)
        img_gd = np.sqrt(img_sobel[:, :, 0]**2 + img_sobel[:, :, 1]**2)
        for i in range(0, img_sobel[:, :, 0].shape[0]):
            for j in range(0, img_sobel[:, :, 0].shape[1]):
                if(np.abs(img_sobel[i, j, 0]) <= 1e-13):
                    img_sobel[i, j, 0] = 1e-12
        tangent = img_sobel[:, :, 1] / img_sobel[:, :, 0]

        # non-maximum suppress
        img_nms = np.zeros(img_gd.shape, img_gd.dtype)
        for i in range(1, h-1):
            for j in range(1, w-1):
                window = img_gd[i-1:i+2, j-1:j+2]
                if(tangent[i, j] >= 1): # linear interpolate for a line
                    gd1 = (1/tangent[i,j]) * window[0, 2] + (1 - (1/tangent[i,j])) * window[0, 1] # Adjacent edges
                    gd2 = (1/tangent[i,j]) * window[2, 0] + (1 - (1/tangent[i,j])) * window[2, 1] # tan(theta) = 1/x , x is edge length
                    if (img_gd[i, j] > gd1 and img_gd[i, j] > gd2):  # get max gd
                        img_nms[i, j] = img_gd[i ,j]
                if (tangent[i, j] <= -1):
                    gd1 = -(1/tangent[i,j]) * window[0, 0] + (1 - (-1 / tangent[i, j])) * window[0, 1]  # tan(pi-theta) = 1/x
                    gd2 = -(1/tangent[i,j]) * window[2, 2] + (1 - (-1 / tangent[i, j])) * window[2, 1]  # pay attention to negative sign
                    if (img_gd[i, j] > gd1 and img_gd[i, j] > gd2):
                        img_nms[i, j] = img_gd[i, j]
                if (tangent[i, j] > 0 and tangent[i, j] < 1):
                    gd1 = (tangent[i,j]) * window[0, 2] + (1 - (tangent[i, j])) * window[1, 2]  # Opposite edge case
                    gd2 = (tangent[i,j]) * window[2, 0] + (1 - (tangent[i, j])) * window[1, 0]  # tan(theta) = x/1
                    if (img_gd[i, j] > gd1 and img_gd[i, j] > gd2):
                        img_nms[i, j] = img_gd[i, j]
                if (tangent[i, j] < 0 and tangent[i, j] > -1):
                    gd1 = -(tangent[i,j]) * window[2, 2] + (1 - (-tangent[i, j])) * window[1, 2]  # pay attention to negative sign
                    gd2 = -(tangent[i,j]) * window[0, 0] + (1 - (-tangent[i, j])) * window[1, 0]  # tan(pi-theta) = x/1
                    if (img_gd[i, j] > gd1 and img_gd[i, j] > gd2):
                        img_nms[i, j] = img_gd[i, j]

        # threshold check and also connect the edge
        stack = list()
        for i in range(1, img_nms.shape[0]-1):
            for j in range(1, img_nms.shape[1]-1):
                if(img_nms[i, j] >= high_boundary):
                    img_nms[i, j] = 255
                    stack.append(tuple((i, j)))
                elif(img_nms[i, j] <= low_boundary):
                    img_nms[i, j] = 0

        # def connectEdgeRecursive(stack, img):
        #     if(len(stack) == 0):
        #         return
        #     (row, col) = stack.pop()
        #     window = img[row - 1:row + 2, col - 1:col + 2]
        #     for i in range(0, window.shape[0]):
        #         for j in range(0, window.shape[1]):
        #             if (window[i, j] < high_boundary and window[i, j] > low_boundary):  # connect the edge
        #                 img[row - 1 + i, col - 1 + j] = 255
        #                 stack.append(tuple((row - 1 + i, col - 1 + j)))
        #     connectEdgeRecursive(stack, img)
        #     return
        # connectEdgeRecursive(stack, img_nms)

        while(len(stack) != 0):
            (row, col) = stack.pop()
            window = img_nms[row-1:row+2, col-1:col+2]
            for i in range(0, window.shape[0]):
                for j in range(0, window.shape[1]):
                    if(window[i, j] < high_boundary and window[i, j] > low_boundary):  # connect the edge
                        img_nms[row-1+i, col-1+j] = 255
                        stack.append(tuple((row-1+i, col-1+j)))

        # delete not 0 and not 255 pixel value
        for i in range(img_nms.shape[0]):
            for j in range(img_nms.shape[1]):
                if img_nms[i, j] != 0 and img_nms[i, j] != 255:
                    img_nms[i, j] = 0

        return img_nms.astype(int)

    @staticmethod
    def getAffineMatrix(originalPixelIndexList: np.ndarray, desiredPixelIndexList: np.ndarray) -> np.ndarray:
        N_original = originalPixelIndexList.shape[0]
        N_desired = desiredPixelIndexList.shape[0]
        if(N_original == N_desired):  # AX=b case , A is square mat
            # x*a11 + y*a12 + a13 + 0*a21 + 0*a22 + 0*a23 - x*X*a31 - X*y*a32 = X
            # 0*a11 + 0*a12 + 0*a13 + x*a21 + y*a22 + a23 - x*Y*a31 - y*Y*a32 = Y
            # need 8 equations constrains for solving 8 vars
            A = np.zeros((8,8), np.float)
            b = np.zeros((8,1), np.float)
            for i in range(0, N_original):
                (x_original, y_original) = originalPixelIndexList[i, :]
                (X_desired, Y_desired) = desiredPixelIndexList[i, :]
                A[2*i:2*i+1+1, :] = np.array([[x_original, y_original, 1, 0, 0, 0, -x_original*X_desired, -y_original*X_desired],
                                              [0, 0, 0, x_original, y_original, 1, -x_original*Y_desired, -y_original*Y_desired]])
                b[2*i:2*i+1+1, 0:1] = np.array([[X_desired],
                                                [Y_desired]])
            # params8 = np.linalg.solve(A, b)
            params8 = np.linalg.pinv(A) @ b
            params9 = np.ones((9,1))
            params9[0:8, 0:1] = params8
            return params9.reshape((3, 3), order='C')
        else:
            print('Fatal error: N_original != N_desired')
            return None

    @staticmethod
    def affineTransfer(img: np.ndarray, affineMat: np.ndarray, desiredImgSize: tuple) -> np. ndarray:
        (h, w) = img.shape[0:2]
        desiredImg = np.zeros(desiredImgSize, img.dtype)
        # relation between (X', Y') and (x, y)
        # (X*a31-a11)*x + (X*a32-a12)*y = a13 - X*a33
        # (Y*a31-a21)*x + (Y*a32 - a22)*y = a23 - Y*a33
        for i in range(0, desiredImgSize[0]):  # here is conner pixel plane in i-j plane
            for j in range(0, desiredImgSize[1]):
                # solve AX=b again
                A = np.array([[i * affineMat[2, 0] - affineMat[0, 0], i * affineMat[2, 1] - affineMat[0, 1]],
                              [j * affineMat[2, 0] - affineMat[1, 0], j * affineMat[2, 1] - affineMat[1, 1]]])
                b = np.array([[affineMat[0, 2] - i * affineMat[2, 2]],
                              [affineMat[1, 2] - j * affineMat[2, 2]]])
                originalImg_i_j_vec = np.linalg.solve(A, b)
                originalImgPixelIndex_i = int(min(max(originalImg_i_j_vec[0, 0], 0), h - 1))
                originalImgPixelIndex_j = int(min(max(originalImg_i_j_vec[1, 0], 0), w - 1))
                originalImgPixelValue = img[originalImgPixelIndex_i, originalImgPixelIndex_j]
                desiredImg[i, j] = originalImgPixelValue
        return desiredImg.astype(int)

        # for i in range(0, h):  # here is conner pixel plane in i-j plane
        #     for j in range(0, w):
        #         pixelValue = img[i, j]
        #         i_j_vec = np.array([[i], [j], [1]])
        #         new_i_j_vec = affineMat @ i_j_vec
        #         # print(new_i_j_vec)
        #         newPixelIndex = (new_i_j_vec[0:2, 0] / new_i_j_vec[2, 0]).astype(int)
        #         # print(newPixelIndex)
        #         newPixelIndex[0] = min(max(newPixelIndex[0], 0), desiredImgSize[0] - 1)
        #         newPixelIndex[1] = min(max(newPixelIndex[1], 0), desiredImgSize[1] - 1)
        #         desiredImg[newPixelIndex[0], newPixelIndex[1]] = pixelValue
        # return desiredImg.astype(int)

        # desiredImg = np.zeros((desiredImgSize[1], desiredImgSize[0]), img.dtype)
        # for x in range(0, w): # here is openCV corner point in x-y (j-i) plane. img[i, j] row-col-order with img[y, x]
        #     for y in range(0, h):
        #         pixelValue = img[y, x]
        #         i_j_vec = np.array([[x], [y], [1]])
        #         new_i_j_vec = affineMat @ i_j_vec
        #         # print(new_i_j_vec)
        #         newPixelIndex = (new_i_j_vec[0:2, 0] / new_i_j_vec[2, 0]).astype(int)
        #         # print(newPixelIndex)
        #         newPixelIndex[0] = min(max(newPixelIndex[0], 0), desiredImgSize[0] - 1)
        #         newPixelIndex[1] = min(max(newPixelIndex[1], 0), desiredImgSize[1] - 1)
        #         desiredImg[newPixelIndex[1], newPixelIndex[0]] = pixelValue
        # return desiredImg.astype(int)

    @staticmethod
    def connerPointDetect(img: np.ndarray, pointsNum: int = 4) -> np. ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
        cnts = cnts[1]  # 如果报错，改回  cnts[0]
        # if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
        docCnt = None

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
            for c in cnts:
                peri = cv2.arcLength(c, True)  # 计算轮廓周长
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
                # 轮廓为4个点表示找到纸张
                if len(approx) == pointsNum:
                    docCnt = approx
                    break
        # for peak in docCnt:
        #     peak = peak[0]
        return docCnt.reshape(pointsNum, 2)

    @staticmethod
    def xyPlane2RowColPlane(xyIndexArr: np.ndarray) -> np.ndarray:
        return np.flip(xyIndexArr, axis=1)

    @staticmethod
    def getClockWise4PixelIndex(rowColIndexArr: np.ndarray) -> np.ndarray:
        max_PixelValue = np.sum(rowColIndexArr[0, :], axis=0)
        max_PixelValueIndex = rowColIndexArr[0, :]
        min_PixelValue = np.sum(rowColIndexArr[0, :], axis=0)
        min_PixelValueIndex = rowColIndexArr[0, :]
        for i in range(1, rowColIndexArr.shape[0]):
            v = np.sum(rowColIndexArr[i, :], axis=0)
            if(v > max_PixelValue):
                max_PixelValue = v
                max_PixelValueIndex = rowColIndexArr[i, :]
            if (v < min_PixelValue):
                min_PixelValue = v
                min_PixelValueIndex = rowColIndexArr[i, :]
        rightConnerLine = list()
        for i in range(0, rowColIndexArr.shape[0]):
            if(rowColIndexArr[i,0]!=min_PixelValueIndex[0] and rowColIndexArr[i,0]!=min_PixelValueIndex[1] and rowColIndexArr[i,0]!=max_PixelValueIndex[0] and rowColIndexArr[i,0]!=max_PixelValueIndex[1]):
                rightConnerLine.append(rowColIndexArr[i, :])
        if(rightConnerLine[0][1] < rightConnerLine[1][1]):
            return np.squeeze(np.array([[min_PixelValueIndex],
                             [rightConnerLine[1]],
                             [max_PixelValueIndex],
                             [rightConnerLine[0]]]), 1)  # assign dim = 1. But if not dim1's shape is 1. error will happen.
        else:
            return np.squeeze(np.array([[min_PixelValueIndex],
                             [rightConnerLine[0]],
                             [max_PixelValueIndex],
                             [rightConnerLine[1]]]))  # auto, or squeeze all shape=1 dim. No error will happen.

    @staticmethod
    def generateClockWise4PixelIndexByImgSize(imgSize: tuple) -> np.ndarray:
        (h, w) = imgSize[0:2]
        return np.array([[0, 0], [0, w], [h, w], [h, 0]])  # i-j plane , img[i,j]

    @staticmethod
    def rotZ(thetaRad, imgShape):
        return np.array([[np.cos(thetaRad), -np.sin(thetaRad), 0],
                        [np.sin(thetaRad), np.cos(thetaRad), 0],
                        [0, 0, 1]])

    @staticmethod
    def rotateImg(img: np.ndarray, thetaRad) -> np.ndarray:
        return ImageUtils.affineTransfer(img, ImageUtils.rotZ(thetaRad, img.shape), img.shape)

    @staticmethod
    def saveImage(img, filename, filepath):
        cv2.imwrite(filepath+filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])