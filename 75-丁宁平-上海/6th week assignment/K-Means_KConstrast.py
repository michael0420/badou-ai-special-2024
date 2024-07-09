# coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.float32类型的N维点集
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

# 获取原始图像
img = cv2.imread('lenna.png')
# print(img.shape)

# 图像二维数据转为一维数据
data = img.reshape((-1, 3))  # reshape(-1,x)矩阵转化为列数为3，行数不指定，由总元素数/列数的值计算而来
print(data)
data = np.float32(data)

# 停止条件(误差 + 最大迭代次数)
cond = (cv2.TERM_CRITERIA_EPS +
        cv2.TermCriteria_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, cond, 10, flags)

# K-Means聚类 聚集成4类
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, cond, 10, flags)

# K-Means聚类 聚集成8类
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, cond, 10, flags)

# K-Means聚类 聚集成16类
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, cond, 10, flags)

# K-Means聚类 聚集成64类
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, cond, 10, flags)

# 图像数据类型转化为uint-8用于显示
centers2 = np.uint8(centers2)
res2 = centers2[labels2.flatten()]
dst2 = res2.reshape((img.shape))

centers4 = np.uint8(centers4)
res4 = centers4[labels4.flatten()]
dst4 = res4.reshape((img.shape))

centers8 = np.uint8(centers8)
res8 = centers8[labels8.flatten()]
dst8 = res8.reshape((img.shape))

centers16 = np.uint8(centers16)
res16 = centers16[labels16.flatten()]
dst16 = res16.reshape((img.shape))

centers64 = np.uint8(centers64)
res64 = centers64[labels64.flatten()]
dst64 = res64.reshape((img.shape))

# 图像转化为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

# 图像显示
plt.show()
