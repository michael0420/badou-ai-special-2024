'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import cv2
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取鸢尾花数据
x, y = load_iris(return_X_y=True)

# pca将4维降成2维
pca = PCA(n_components=2)
pca.fit(x)
newx = pca.fit_transform(x)
newx = np.float32(newx)
# print(newx)

# 原数据降维后的分类
r_x, r_y = [], []
g_x, g_y = [], []
b_x, b_y = [], []
for i in range(len(y)):
    if y[i] == 0:
        r_x.append(newx[i][0])
        r_y.append(newx[i][1])

    elif y[i] == 1:
        g_x.append(newx[i][0])
        g_y.append(newx[i][1])

    else:
        b_x.append(newx[i][0])
        b_y.append(newx[i][1])

# 利用k-means对降维后的数据进行分类
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

retval, bestLabels, centers = cv2.kmeans(newx, K, None, criteria, attempts, flags)
print(retval)
print(bestLabels)
print(centers)

r_a, r_b = [], []
g_a, g_b = [], []
b_a, b_b = [], []
for i in range(len(bestLabels)):
    if bestLabels[i] == 0:
        r_a.append(newx[i][0])
        r_b.append(newx[i][1])

    elif bestLabels[i] == 1:
        g_a.append(newx[i][0])
        g_b.append(newx[i][1])

    else:
        b_a.append(newx[i][0])
        b_b.append(newx[i][1])

# 对比分类结果
plt.subplot(121)
plt.title("k_means")
plt.scatter(r_a, r_b, c='b', marker='x')
plt.scatter(g_a, g_b, c='g', marker='x')
plt.scatter(b_a, b_b, c='r', marker='x')

plt.subplot(122)
plt.title("pca")
plt.scatter(r_x, r_y, c='r', marker='x')
plt.scatter(g_x, g_y, c='g', marker='x')
plt.scatter(b_x, b_y, c='b', marker='x')

plt.show()
