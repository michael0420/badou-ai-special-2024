# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import KMeansUtil as KMUitl

"""
第六周作业
1. 实现k-means
"""
gray_lenna = cv2.imread('../Image/lenna.png', 0)
kmeans_gray_lenna = KMUitl.image_kmeans(gray_lenna, 4, 10)
KMUitl.show_kmeans_results(['gray', 'kmeans gray'], [gray_lenna, kmeans_gray_lenna])

# 聚类越多，越接近原图像，至少聚类到一定数量类别，人眼难以分辨原图与聚类后的图
lenna = cv2.imread('../Image/lenna.png')
kmeans_lenna2 = KMUitl.image_kmeans(lenna, 2, 10)
kmeans_lenna4 = KMUitl.image_kmeans(lenna, 4, 10)
kmeans_lenna8 = KMUitl.image_kmeans(lenna, 8, 10)
kmeans_lenna16 = KMUitl.image_kmeans(lenna, 16, 10)
titles = ['lenna', 'lenna2', 'lenna4', 'lenna8', 'lenna16']
images = [cv2.cvtColor(lenna, cv2.COLOR_BGR2RGB), kmeans_lenna2, kmeans_lenna4, kmeans_lenna8, kmeans_lenna16]
KMUitl.show_kmeans_results(titles, images)

# athlete
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
predict = KMUitl.kmeans_athlete(X, 3)
KMUitl.show_kmeans_predict(X, predict)
