# coding: utf-8
import os

#current_directory = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
# parent_directory = os.path.dirname(current_directory)  # 获取上一层目录的路径
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
输出参数：
1.compactness：紧密度，返回每个点到相应重心的距离的平方和。
2.labels：标志数组（与上一节提到的代码相同），每个成员被标记为 0，1 等
3.centers：由聚类的中心组成的数组
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)
print (img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:]

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签，KMEANS_RANDOM_CENTERS，随机选取中心点
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
print("labels:\n",labels)
print("centers:\n",centers)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
print("dst:\n",dst)

# 用来正常显示中文标签：利用matplotlib绘图中文字符显示为方框的乱码问题
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   # plt.xticks([]),plt.yticks([])  不显示X轴，Y轴
plt.show()

"""
current_directory = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
# parent_directory = os.path.dirname(current_directory)  # 获取上一层目录的路径
# 原图路径
img_path = current_directory + "\lenna.png"
# 读图
img = cv2.imread(img_path,0)
"""