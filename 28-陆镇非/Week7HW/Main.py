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
from HierarchicalCluster import *
from DensityCluster import *
from Sift import *
from sklearn.datasets import load_iris
np.set_printoptions(threshold=np.inf)


class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test13(self):
        X = np.random.random((50, 2))
        # X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
        X = np.array(X)
        hc = HierarchicalCluster()
        hist, root, nodeList = hc.fit(X)
        print(hist)
        print(hist.shape)
        hc.plotDendrogramByThirdPartyLib(hist)
        root.drawTreeBFS()
        root.drawTreeDFS(0, 0, fontsize=8, isRoot=True)
        root.printTreeBFS()
        # root.printTreeDFS()

        distance_threshold = hist[hist.shape[0]-1-5, 2]
        # distance_threshold = 1.5
        print("distance_threshold = ", distance_threshold)
        dict_index_typeCenter = hc.getTypesBydistance(distance_threshold, hist, X)
        print(dict_index_typeCenter)
        typeLabel = np.array([dict_index_typeCenter[i][0] for i in range(0, len(dict_index_typeCenter))])
        print(typeLabel)

        plt.figure()
        plt.title("HierarchicalCluster")
        plt.scatter(X[:, 0], X[:, 1], c=typeLabel, marker='o')

        for k, v in dict_index_typeCenter.items():
            plt.scatter(v[1][0], v[1][1], c=v[0], marker='x')


    def test14(self):
        # X = np.random.random((120, 2))
        iris = load_iris()
        X = iris.data[:, :4]  # take first 4 column-dim features  . it's a 4-dim data (not 2 dim x-y data)
        print(X.shape)
        dc = DensityCluster(eps=0.42, minPts=8)
        dict_index_type, DObjects = dc.fit(X)
        print(dict_index_type)
        for i in range(len(DObjects)):
            print(f"type: {i}, count: {len(DObjects[i])}")
        type = np.array([dict_index_type[i] for i in range(len(dict_index_type))])  # have to be np.ndarray type
        print(type)
        x0 = X[type == 0]  # have to be np.ndarray type to make compare ==0 , it will return a ndarray true or false matrix
        x1 = X[type == 1]
        x2 = X[type == 2]
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

    def test15(self):  # sift
        imageFilePath1 = "./iphone1.png"
        BGRImage1 = ImageUtils.readImgFile2BGRImage(imageFilePath1)
        greyImage1 = ImageUtils.BGRImage2GreyImage(BGRImage1)
        s = Sift()
        keypoints1, descriptor1 = s.detect(greyImage1)
        s.drawKeyPoints(keypoints1, BGRImage1, "Image1")

        imageFilePath2 = "./iphone2.png"
        BGRImage2 = ImageUtils.readImgFile2BGRImage(imageFilePath2)
        greyImage2 = ImageUtils.BGRImage2GreyImage(BGRImage2)
        keypoints2, descriptor2 = s.detect(greyImage2)
        s.drawKeyPoints(keypoints2, BGRImage2, "Image2")

        goodMatch = s.descriptorsMatch_CV(descriptor1, descriptor2)
        s.drawMatches_my_ij(BGRImage1, keypoints1, BGRImage2, keypoints2, goodMatch[:20], "my")

        # openCV existed sift:
        imageFilePath1 = "./iphone1.png"
        BGRImage1_ = ImageUtils.readImgFile2BGRImage(imageFilePath1)
        greyImage1_ = ImageUtils.BGRImage2GreyImage(BGRImage1_)
        imageFilePath2 = "./iphone2.png"
        BGRImage2_ = ImageUtils.readImgFile2BGRImage(imageFilePath2)
        greyImage2_ = ImageUtils.BGRImage2GreyImage(BGRImage2_)
        keypoints1_, descriptor1_ = s.detect_CV(greyImage1_)
        keypoints2_, descriptor2_ = s.detect_CV(greyImage2_)
        s.drawKeyPoints_CV(keypoints1_, BGRImage1_, "Image1_")
        s.drawKeyPoints_CV(keypoints2_, BGRImage2_, "Image2_")
        goodMatch_ = s.descriptorsMatch_CV(descriptor1_, descriptor2_)
        s.drawMatches_CV_xy(BGRImage1_, keypoints1_, BGRImage2_, keypoints2_, goodMatch_[:20], "CV")

        cv2.waitKey(0)

    def runAlltests(self) -> None:
        # test13
        start_time = time.time()
        self.test13()
        end_time = time.time()
        print("test13 excuted time cost：", end_time - start_time, "seconds")

        # test14
        start_time = time.time()
        self.test14()
        end_time = time.time()
        print("test14 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")

        # test15
        start_time = time.time()
        self.test15()
        end_time = time.time()
        print("test15 excuted time cost：", end_time - start_time, "seconds")


if __name__ == "__main__":
    solution = Solution()
