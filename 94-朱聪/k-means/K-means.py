# coding: utf-8

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
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)
print(img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:] # 灰度图只有2个值了，没有通道数了

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon) 停止条件为达到最大迭代次数或者达到指定的精度。10表示最大迭代次数为10次，1.0表示精度为1.0
# 最大迭代次数：超过10次迭代仍未达到预设的精度，则停止迭代
# 精度，即如果K-means算法执行一次迭代后得到的聚类结果与上一次迭代相比的紧凑度变化量小于1.0，则停止迭代
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签，表示使用随机初始化的方式来选择初始聚类中心
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 data是要聚类的数据，4表示聚类的类别数目，None表示没有预设的聚类中心，criteria和10表示停止条件和最大迭代次数，flags表示聚类中心的初始化方式
# 结果包括紧凑度(compactness)、样本点所属的类别标签(labels)和聚类中心(centers)。
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
# labels.flatten()
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([]) # plt.xticks([])和plt.yticks([])隐藏坐标轴刻度线
plt.show()