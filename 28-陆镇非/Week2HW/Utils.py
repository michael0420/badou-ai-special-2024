import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageUtils(object):
    @staticmethod
    def readImgFile2BGRImage(filePath: str) -> np.ndarray:
        img = cv2.imread(filePath)
        return img

    @staticmethod
    def plotAllImages(imgDict: dict, showImmediately:bool = True) -> None:
        N = len(imgDict)
        plt.figure()
        i = 1
        for key, value in imgDict.items():
            plt.subplot(1, N, i)
            plt.imshow(value, cmap='gray')
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

    @staticmethod
    def greyImage2BinaryImage(greyImage: np.ndarray) -> np.ndarray:
        (h, w) = greyImage.shape
        BinaryImage = np.zeros((h, w), 'float')
        for i in range(0, h):
            for j in range(0, w):
                val = greyImage[i, j] / 255
                if(val > 0.5):
                    BinaryImage[i, j] = 1
                else:
                    BinaryImage[i, j] = 0
        return BinaryImage

    @staticmethod
    def BGRImageNearestInteropate(img: np.ndarray, targetImgSize: np.ndarray) -> np.ndarray:
        (h, w, channels) = img.shape[0:3]
        (h_target, w_target) = targetImgSize[0:2]
        h_stepSize = h / h_target
        w_stepSize = w / w_target
        targetImage = np.zeros((h_target, w_target, channels), img.dtype)

        for i in range(0, h_target):
            for j in range(0, w_target):
                original_pixel_index_i = int(h_stepSize * i + 0.5)
                original_pixel_index_j = int(w_stepSize * j + 0.5)
                # print(i, j, original_pixel_index_i, original_pixel_index_j, h , w)
                if(original_pixel_index_i > h - 1):  # more robust for index exceeding the max index
                    original_pixel_index_i = h - 1
                if (original_pixel_index_j > w - 1):
                    original_pixel_index_j = w - 1
                targetImage[i, j, :] = img[original_pixel_index_i, original_pixel_index_j, :]
        return targetImage

