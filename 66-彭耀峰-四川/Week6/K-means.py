'''
将灰度图聚类为4类
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0)
rows, cols = img.shape

# 转为一维，聚类数据最好是np.flloat32类型的N维点集
data = img.reshape(rows * cols, 1)
data = np.float32(data)

# 设置停止条件 (type,max_iter,epsilon)
# —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
# —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
# —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 初始中心选择方式，两种：KMEANS_PP_CENTERS、KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

# K-means聚类，聚成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

plt.figure(1)
plt.imshow(dst, 'gray')
plt.show()
