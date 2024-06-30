# -*- coding: utf-8 -*-
'''
@File    :   kmeans.py
@Time    :   2024/05/18 20:16:07
@Author  :   廖红洋 
'''
import cv2
import numpy as np

img = cv2.imread("lenna.png",0)#为0表示读取灰度图
h,w=img.shape[:]
k=6#设置K值为6，聚类为6类
data = np.float32(img.reshape((h * w, 1)))

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS#设置随机初始点
retv, mask, centers = cv2.kmeans(data , k, None, criteria, 5, flags)
final = mask.reshape((h, w))
final = (final*255/k).astype(np.uint8)#均衡化
print(final)
cv2.imshow('result',final)
cv2.waitKey(0)