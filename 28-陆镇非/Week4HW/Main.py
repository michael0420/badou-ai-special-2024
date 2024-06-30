# Author: Zhenfei Lu
# Created Date: 4/15/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import time
from Utils import ImageUtils
from Noise import *
from sklearn.datasets import load_iris
from PCA import *
import matplotlib.pyplot as plt


class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test6(self) -> None:
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        greyImage = ImageUtils.BGRImage2GreyImage(BGRImage)
        gaussianNoise = GaussianNoise(percentage=0.9, mu=5, sigma=10)
        gN_img = gaussianNoise.addNoise4Img(greyImage)
        pepperSaltNoise = PepperSaltNoise(percentage=0.8)
        ps_img = pepperSaltNoise.addNoise4Img(greyImage)
        poissonNoise = PoissonNoise(percentage=0.8, lamb=20)
        pois_img = poissonNoise.addNoise4Img(greyImage)
        dict2 = {}
        dict2['grey image' ] = greyImage
        dict2['gaussian' ] = gN_img
        dict2['pepperSalt'] = ps_img
        dict2['poisson'] = pois_img
        ImageUtils.plotAllRGBImages(dict2, False)
        return
        # skiimage existed libs:
        # from skimage import util
        # noise_gs_img = util.random_noise(img, mode='poisson')

    def test7(self) -> None:
        x, y = load_iris(return_X_y=True)  # load data, x is feature (150, 4)  y is label (150, )
        print(x.shape, y.shape)
        pca = PCA(features=x, targetDimension=2)
        reducedDimFeature = pca.reduceFeatureDim()
        print(reducedDimFeature.shape)

        def plot2DimFeatures(x: np.ndarray, y: np.ndarray, title: str) -> None:
            N = x.shape[0]
            redDatax, redDatay = list(), list()
            greenDatax, greenDatay = list(), list()
            blueDatax, blueDatay = list(), list()
            for i in range(0, N):
                if (y[i] == 0):
                    redDatax.append(x[i, 0])
                    redDatay.append(x[i, 1])
                elif (y[i] == 1):
                    greenDatax.append(x[i, 0])
                    greenDatay.append(x[i, 1])
                elif (y[i] == 2):
                    blueDatax.append(x[i, 0])
                    blueDatay.append(x[i, 1])
            plt.figure()
            plt.title(title)
            plt.scatter(redDatax, redDatay, c='r', marker='x')
            plt.scatter(greenDatax, greenDatay, c='g', marker='.')
            plt.scatter(blueDatax, blueDatay, c='b', marker='D')

        # reducedDimFeature plot
        plot2DimFeatures(reducedDimFeature, y, 'reduced Dim Feature')
        # original data[0:2] plot (just choose first 2 dim, not all)
        plot2DimFeatures(x, y, 'original Feature')
        return
        # sklearn existed libs:
        # import sklearn.decomposition as dp
        # x, y = load_iris(return_X_y=True)
        # pca = dp.PCA(n_components=2)
        # reduced_x = pca.fit_transform(x)

    def runAlltests(self) -> None:
        # test6
        start_time = time.time()
        self.test6()
        end_time = time.time()
        print("test6 excuted time cost：", end_time - start_time, "seconds")

        # test7
        start_time = time.time()
        self.test7()
        end_time = time.time()
        print("test7 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")

if __name__ == "__main__":
    solution = Solution()
