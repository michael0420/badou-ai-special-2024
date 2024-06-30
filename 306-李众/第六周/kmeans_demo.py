# _*_ coding: UTF-8 _*_
import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('../data/lenna.png')
    gray = np.zeros((img.shape[0], img.shape[1]), img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            m = img[i, j]
            gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.33)
    H, W = gray.shape[:]
    print(H, W)
    data = gray.reshape((H * W, 1))
    data = np.float32(data)

    # 设置停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签(初始中心随机选择)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 利用opencv中kmeans聚类成4类
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

    dst = labels.reshape((gray.shape[0], gray.shape[1]))

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    images = [gray, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
