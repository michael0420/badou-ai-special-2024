# coding = utf-8

'''
        实现灰度图K-Means聚类
'''

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

img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray_img)

# img = cv2.imread('lenna.png', 0)        # 读取图像灰度值
print(gray_img.shape)
w, h = gray_img.shape[:]        # 获取灰度图宽高

# 将图像数据转换为一维
one = gray_img.reshape((w * h), 1)
one = np.float32(one)

# 设置终止模式
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 0)
# 开始聚类
# labels数组，每个元素是数据点所属集群的标签（从 0 到 K-1）
# centers数组，包含每个集群的最终质心位置
_, labels, centers = cv2.kmeans(one, 2, None, criteria, 8, cv2.KMEANS_USE_INITIAL_LABELS)
print(centers)

# 生成最终图像，重新调整为二维图像
res = labels.reshape((gray_img.shape[0], gray_img.shape[1]))
res = cv2.convertScaleAbs(res)      # openCV需要把图像数据转换为8位无符号整数类型再显示
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
# cv2.imshow('k-means', res)
# cv2.waitKey(0)
titles = [u'原始图像', u'聚类图像']
images = [gray_img, res]
for i in range(2):
   plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]), plt.yticks([])
plt.show()
