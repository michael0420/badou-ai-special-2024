# Author: Zhenfei Lu
# Created Date: 5/18/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(999999999)
from KMeans import *

class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test11(self):
        X = [[0.0888, 0.5885],
             [0.1399, 0.8291],
             [0.0747, 0.4974],
             [0.0983, 0.5772],
             [0.1276, 0.5703],
             [0.1671, 0.5835],
             [0.1306, 0.5276],
             [0.1061, 0.5523],
             [0.2446, 0.4007],
             [0.1670, 0.4770],
             [0.2485, 0.4313],
             [0.1227, 0.4909],
             [0.1240, 0.5668],
             [0.1461, 0.5113],
             [0.2315, 0.3788],
             [0.0494, 0.5590],
             [0.1107, 0.4799],
             [0.1121, 0.5735],
             [0.1007, 0.6318],
             [0.2567, 0.4326],
             [0.1956, 0.4280]
             ]
        # X = np.array(X)
        # print(X.shape)
        X = np.random.random((1000, 2))
        clusters = 3
        kmeans = KMeans(clusters)
        Kobjects, dict_index_type = kmeans.fit(np.array(X), 50, 1e-5, batch_size=X.shape[0])

        plt.figure()
        plt.scatter([x[0] for x in X], [x[1] for x in X], c=[value for key, value in dict_index_type.items()], marker='o')

        for i in range(0, len(Kobjects)):
            print(Kobjects[i].center)
            print(Kobjects[i].children)
            print("---------------------")
            plt.scatter(Kobjects[i].center[0,], Kobjects[i].center[1,], c=i, marker='x')
        print(dict_index_type)
        # existed lib
        # from sklearn.cluster import KMeans
        # clf = KMeans(n_clusters=3)
        # y_pred = clf.fit_predict(X)

    def test12(self):
        filePath = "./lenna.png"
        img = cv2.imread(filePath)  # openCV existed lib
        greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clusters = 4
        kmeans = KMeans(clusters)
        kmeansImg = kmeans.fit4image(greyImg, epoches=50, metric=1, batch_size=greyImg.shape[0])
        plt.figure()
        plt.imshow(kmeansImg, cmap='gray')

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # openCV existed lib
        kmeansImg = kmeans.fit4image(imgRGB, epoches=50, metric=1, batch_size=imgRGB.shape[0])
        plt.figure()
        plt.imshow(kmeansImg, cmap='gray')
        # openCV existed lib
        # compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

    def runAlltests(self) -> None:
        # test11
        start_time = time.time()
        self.test11()
        end_time = time.time()
        print("test11 excuted time cost：", end_time - start_time, "seconds")

        # test12
        start_time = time.time()
        self.test12()
        end_time = time.time()
        print("test12 excuted time cost：", end_time - start_time, "seconds")

        plt.show()
        print("All plots shown")


if __name__ == "__main__":
    solution = Solution()
