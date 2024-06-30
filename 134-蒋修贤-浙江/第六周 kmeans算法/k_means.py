#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.jpeg',0)

rows, cols = img.shape[:]
data = np.float32(img.reshape((rows*cols,1)))
critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels,centers = cv2.kmeans(data,4,None,critera,10,flags)

dst = labels.reshape((img.shape[0],img.shape[1]))
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.subplot(1,2,1)
plt.imshow(img,'gray')
plt.subplot(1,2,2)
plt.imshow(dst,'gray')
plt.show()