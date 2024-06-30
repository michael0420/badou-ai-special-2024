# -*- coding: utf-8 -*-
"""
@author: zhjd

Use KMeans clustering algorithm to achieve image compression
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_compress(img, k):
    """

    """
    data = img.reshape((-1, len(img.shape)))
    data = np.float32(data)
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data, k, None, kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape(img.shape)
    return cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    img = cv2.imread('../alex.jpg')
    dst4 = image_compress(img, 4)
    dst8 = image_compress(img, 8)
    dst16 = image_compress(img, 16)
    dst32 = image_compress(img, 32)
    dst64 = image_compress(img, 64)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像 k=4', u'聚类图像 k=8', u'聚类图像 k=16', u'聚类图像 k=32', u'聚类图像 k=64']
    images = [img, dst4, dst8, dst16, dst32, dst64]
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
