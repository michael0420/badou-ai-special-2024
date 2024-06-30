'''
采用Kmeans进行彩色图像分割
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lenna.png')
h, w, c = img.shape
# 每个通道内的二维数据改为一维，并进行float32格式化
data = img.reshape([h * w, c])
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # max_iter迭代次数，epsilon精确度
flags = cv2.KMEANS_RANDOM_CENTERS

compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags, None)
# print(compactness2)  # 所有样本点与中心点距离和
# print(data, data.shape)
# print(labels2, labels2.shape)  # 所有点的聚类标签
# print(centers2)  # 聚类中心点
centers2 = np.uint8(centers2)
dst = centers2[labels2.flatten()]
'''
centers2的2个中心点按0,1序号，赋值给labels2.flatten中对应的值
A = np.array([[4, 2],
             [2, 3]])
B = [0, 1, 1, 0, 1, 0, 1, 1]
C = A[B]
print(C) = [[4 2]
            [2 3]
            [2 3]
            [4 2]
            [2 3]
            [4 2]
            [2 3]
            [2 3]]
'''
dst = dst.reshape([h, w, c])
cv2.imshow('1', dst)
cv2.waitKey()

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = []
for i in range(1, 7):
    K = 2 * i
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags, None)
    centers = np.uint8(centers)
    dst = centers[labels.flatten()]
    dst = dst.reshape([h, w, c])
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i), plt.xticks([]), plt.yticks([])
    plt.title('聚类阶次为{}'.format(K), loc='center')
    plt.imshow(dst, cmap='gray')
plt.show()
















