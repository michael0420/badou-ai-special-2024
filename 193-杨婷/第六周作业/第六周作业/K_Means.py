import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0)  # 读取原始图像灰度图
# print(img.shape)
print(img)
rows, cols = img.shape

# 图像二维像素转换为一维
data = img.reshape((rows*cols, 1))
data = np.float32(data)  # data表示聚类数据，最好是np.float32类型的N维点集
print(data)

# 停止条件 (type,max_iter最大迭代次数,epsilon精确度)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
# 随机选择初始质心，而cv2.KMEANS_PP_CENTERS是一种基于K-Means++算法的初始化策略


'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, None, criteria, attempts, flags[, centers])
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
# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)  # 这里None表示初始聚类中心由设置的flags选择
dst = labels.reshape((rows, cols))
print(dst)
print('---------------')
print(compactness)  # 所有数据点到其对应聚类中心的距离的平方和。这个值越小，表示聚类效果越好。
print('---------------')
print(centers)   # 每一行代表一个聚类中心的坐标。这个数组的大小是k

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = ['原始图像', '聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # 用于设置x轴和y轴的刻度,在这里不显示x轴和y轴的刻度
plt.show()
