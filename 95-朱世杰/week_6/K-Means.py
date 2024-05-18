"""
实现K-Means聚类
@AuThor：zsj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../lenna.png", 0)
row, col = img.shape

# 将二维灰度图转成一维
data = img.reshape((row * col, 1))
data = np.float32(data)

# —————— OpenCV中的K-Means聚类 ——————
# cv.kmeans()函数进行数据聚类：
#   sample：它应该是**np.float32**数据类型
#   nclusters(K)：结束条件所需的簇数
#   criteria：这是迭代终止条件。满足此条件后，算法迭代将停止。实际上，它应该是3个参数的元组。它们是(type,max_iter,epsilon)： a. 终止条件的类型。它具有3个标志，如下所示：
#       cv.TERM_CRITERIA_EPS-如果达到指定的精度epsilon，则停止算法迭代。
#       cv.TERM_CRITERIA_MAX_ITER-在指定的迭代次数max_iter之后停止算法。
#       cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER-当满足上述任何条件时，停止迭代。
# 输出参数：
#   紧凑度：它是每个点到其相应中心的平方距离的总和。
#   标签：这是标签数组（与上一篇文章中的“代码”相同），其中每个元素标记为“0”，“ 1” .....
#   中心：这是群集中心的阵列。
K = 4
# 每当运行10次算法迭代或达到epsilon = 1.0的精度时，就停止算法并返回答案。
# 定义终止标准 = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置标志
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)

tar = labels.reshape((img.shape))

# 展示图片
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
imgs = [img, tar]
titles = ['原始图像', '聚类图象']
length = len(imgs)
for i in range(length):
    plt.subplot(1, length, i + 1)
    plt.title(titles[i])
    plt.imshow(imgs[i], cmap="gray")
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
plt.show()

colors = ['r', 'b', 'y', 'g']
for i in range(K):
    kind = data[labels == i]
    print(kind)
    plt.hist(kind, 256, [0, 256], color=colors[i])
expanded_centers = [center for center in centers.flatten() for _ in range(1000)]
plt.hist(expanded_centers, 256, [0, 256], color='k')
plt.show()
