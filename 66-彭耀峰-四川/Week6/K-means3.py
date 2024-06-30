'''
将彩色图像聚类为3类
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')
# 将二维像素转为一维，聚类数据最好是np.flloat32类型的N维点集
data = img.reshape((-1,3))
data = np.float32(data)
print(data)

# 设置停止条件(type,max_iter,epsilon)
# —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
# —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
# —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 初始中心选择方式，两种：KMEANS_PP_CENTERS、KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成3类
compactness, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, flags)

#图像转换回uint8二维类型
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape((img.shape))
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.imshow(dst,'gray')
plt.show()