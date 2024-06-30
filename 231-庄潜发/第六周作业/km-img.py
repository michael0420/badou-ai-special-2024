"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/15 23:14
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
# 灰度图
img = cv2.imread("../lenna.png", flags=0)
# 转为一维, 并且类型转为float32
h, w = img.shape[:2]
data_X = img.reshape((h*w, 1))
data = np.float32(data_X)
K = 4
bestLabels = None
# 迭代十次, 误差为1.0, 这两个条件任一符合终止迭代
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签, 上面设置了迭代为10次
attempts = 10
# 设置标签, 随机获取质心
flags = cv2.KMEANS_RANDOM_CENTERS
# label为聚类后的一维张量, 将其转换为二维即可得到聚类后的类别
retval, label, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)
# 转换为二维
dst_img = label.reshape((h, w))
# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示聚类后的图像
plt.imshow(dst_img, 'gray')  # 将得到的像素点聚类类别这些数值映射到灰度级上
plt.title(u'聚类图像')
plt.xticks([]),plt.yticks([])  # 去除坐标印记
plt.show()
















