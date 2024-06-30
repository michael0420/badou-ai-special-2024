# Author: Zhenfei Lu
# Created Date: 6/15/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import time
from Utils import ImageUtils
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(999999999)
from KMeans import *
from HierarchicalCluster import *
from DensityCluster import *
from RegressionModel import *
from Ransac import *
from Hash import *
np.set_printoptions(threshold=np.inf)


class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test16(self):
        # Fully connected model solved by nonLinear sqr numerical method
        x_train = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y_train = np.sin(x_train).reshape(-1, 1) + np.random.normal(scale=0.1, size=x_train.shape)
        fcModel = FullyConnectedModel()
        optimizedParams = fcModel.fit(X=x_train, Y=y_train, epoch=5000)
        print(optimizedParams)
        y_predict = fcModel(optimizedParams, x_train)

        plt.figure()
        plt.scatter(x_train, y_train, label='data')
        plt.plot(x_train, y_predict, 'r', label='fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('truth vs fit')
        plt.legend()

    def test17(self):
        # add inner data
        a0 = 2.19
        a1 = 3.24
        x_train = np.linspace(0, 5, 500).reshape(-1, 1)
        y_train = a0 + a1 * x_train + 5*np.random.random((500, 1))
        # add outer dater
        x_noise = np.linspace(0, 5, 100).reshape(-1, 1)
        y_noise = -2*a1 + 0 * x_noise + 8*np.random.random((100, 1))

        x_noise_more = np.linspace(0, 5, 30).reshape(-1, 1)
        y_noise_more = 22 * np.random.random((30, 1)) + 3

        X = np.vstack((x_train, x_noise, x_noise_more))
        Y = np.vstack((y_train, y_noise, y_noise_more))
        linearModel = LinearModel()
        optimizedParams, loss = linearModel.fit(X=X, Y=Y, epoch=5000)
        print(optimizedParams)
        y_predict = linearModel(optimizedParams, X)
        plt.figure()
        plt.scatter(X, Y, label='All data with noise')
        plt.plot(X, y_predict, 'r', label='linear-leastsqr fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('truth vs fit vs data')

        linearModel = LinearModel()
        ransac = Ransac(linearModel, X, Y)
        bestModel, interiorDataIndex, loss = ransac.fit(num_forFit=100, maxIter=500, L2NormLossThreshold=5.0, num_interiorData=300)
        # print(loss)
        # print(linearModel, bestModel)
        y_predict = bestModel(bestModel.params, X)
        X_interior = X[interiorDataIndex]
        Y_interior = Y[interiorDataIndex]
        plt.plot(X, y_predict, 'g', label='ransac fit')
        plt.scatter(X_interior, Y_interior, c="black", marker='x', label='ransac interior data')
        plt.legend()

        # also use density cluster run
        dataPack = np.hstack((X, Y))
        dc = DensityCluster(eps=0.93, minPts=2)
        dict_index_type, DObjects = dc.fit(dataPack)
        print(dict_index_type)
        for i in range(len(DObjects)):
            print(f"type: {i}, count: {len(DObjects[i])}")
        type = np.array([dict_index_type[i] for i in range(len(dict_index_type))])  # have to be np.ndarray type
        print(type)
        x0 = dataPack[type == 0]  # have to be np.ndarray type to make compare ==0 , it will return a ndarray true or false matrix
        x1 = dataPack[type == 1]
        x2 = dataPack[type == 2]
        # x3 = X[type == -1]  # noise type
        plt.figure()
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='type0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='type1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='type2')
        # plt.scatter(x3[:, 0], x3[:, 1], c="black", marker='x', label='noise')
        plt.title("DensityCluster")
        plt.xlabel('x value')
        plt.ylabel('y value')
        plt.legend(loc=1)

        # use Kmeans run
        clusters = 3
        kmeans = KMeans(clusters)
        Kobjects, dict_index_type = kmeans.fit(np.array(dataPack), 50, 1e-5, batch_size=dataPack.shape[0], shuffling=True)
        plt.figure()
        plt.scatter([x[0] for x in dataPack], [x[1] for x in dataPack], c=[dict_index_type[i] for i in range(len(dict_index_type))], marker='o')
        for i in range(0, len(Kobjects)):
            plt.scatter(Kobjects[i].center[0,], Kobjects[i].center[1,], c=i, marker='x')
        # use Hierarchical cluster run too slow

    def test18(self):
        imageFilePath = "./lenna.png"
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        gaussianFilterKernel = ImageUtils.sobelX
        img_new = np.zeros(BGRImage.shape)
        for i in range(img_new.shape[2]):
            img_new[:,:,i] = ImageUtils.convolutionFilterC1(BGRImage[:,:,i], gaussianFilterKernel)
        ImageUtils.saveImage(img_new, "lenna_noise.png", "./")
        phash1 = Hash.getPHash(BGRImage)
        phash2 = Hash.getPHash(img_new)
        result = Hash.compareTwoHashStr(phash1, phash2)
        print("phash: ", result)
        phash1 = Hash.getAverageHash(BGRImage)
        phash2 = Hash.getAverageHash(img_new)
        result = Hash.compareTwoHashStr(phash1, phash2)
        print("ahash: ", result)
        phash1 = Hash.getDifferenceHash(BGRImage)
        phash2 = Hash.getDifferenceHash(img_new)
        result = Hash.compareTwoHashStr(phash1, phash2)
        print("dhash: ", result)

    def runAlltests(self) -> None:
        # test16
        start_time = time.time()
        self.test16()
        end_time = time.time()
        print("test16 excuted time cost：", end_time - start_time, "seconds")

        # test17
        start_time = time.time()
        self.test17()
        end_time = time.time()
        print("test17 excuted time cost：", end_time - start_time, "seconds")

        # test18
        start_time = time.time()
        self.test18()
        end_time = time.time()
        print("test18 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")


if __name__ == "__main__":
    solution = Solution()
