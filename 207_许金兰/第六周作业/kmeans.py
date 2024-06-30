"""
@author: 207-xujinlan
kmeans处理图片
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1.读取灰度图片
pic_path = 'lenna.png'
img = cv2.imread(pic_path, 0)
# 2.将数据从二维转成一维
w, h = img.shape
data = np.float32(img.reshape((w * h, 1)))
# 3.设置聚类条件
# 设置迭代停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置初始中心点选择方式
flags = cv2.KMEANS_RANDOM_CENTERS
# 4.聚类计算
compactness, labels, centers = cv2.kmeans(data, 10, None, criteria, 10, flags)
# 5.将一维数据转换成二维数据
dst_img = labels.reshape((img.shape[0], img.shape[1]))
# 6.展示图片
plt.figure(1)
plt.title('source img')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.figure(2)
plt.title('dst img')
plt.imshow(dst_img, cmap='gray')
plt.axis('off')
