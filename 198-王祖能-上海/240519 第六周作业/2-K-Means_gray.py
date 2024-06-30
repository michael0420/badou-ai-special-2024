'''
采用cv2库Kmeans聚类进行图像分割
'''
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
在OpenCV中，Kmeans()函数原型如下所示：
compactness, labels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    compactness:紧密度，返回每个点到相应重心的距离的平方和。
    labels：标志数组（与上一节提到的代码相同），每个成员被标记为0，1等。
    centers：由聚类的中心组成的数组。   
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类簇数
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
img = cv2.imread('lenna.png', 0)
h, w = img.shape
# 图像矩阵数据改为1列，并改为float32数据格式进行cv2.kmeans处理
data = img.reshape([h * w, 1])
data = np.float32(data)
'''
cv2.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])应用提前赋值各个变量
'''
# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # max_iter = 10, epsilon = 1.0
# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
# 数据聚类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
# 变为正常 h * w 图像矩阵
dst = labels.reshape(h, w)
print(dst)

# 设置可以正常显示中文字体
plt.ion()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure('method1')
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('原始图像', loc='left'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(dst, cmap='gray'), plt.title(u'聚类分割图像'), plt.xticks([]), plt.yticks([])
# 简便的方法来隐藏刻度标签，使得 x/y 轴上不显示任何刻度标签。
plt.show()
plt.pause(2)
# 老师的循环方法
images = [img, dst]
titles = ['原始图像', '聚类分割图像']
locations = ['left', 'right']
plt.figure('method2')
for i in range(2):
    plt.subplot(1, 2, i+1), plt.title(titles[i], loc=locations[i])
    plt.imshow(images[i], cmap='gray')
    plt.xticks([]), plt.yticks()
plt.show()
plt.pause(2)
plt.close()
