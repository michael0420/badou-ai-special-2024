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

#读取原始图像灰度颜色
img=cv2.imread('lenna.png',0)
print(img.shape)

#获取图像高度、宽度
# rows, cols = img.shape[:]
high=img.shape[0]
width=img.shape[1]
#图像二维像素转换为一维
data=img.reshape((high*width,1))
# data1=img.reshape((-1,1))
data=np.float32(data)
# data1=np.float32(data1)
print(data)
# print(data1)

#停止条件 (type,max_iter,epsilon)
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)   # 迭代次数和精度设置

#设置标签
flags=cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness,labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)

#重新生成二维图像
dst=labels.reshape((high,width))
print(labels)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
# titles = [u'原始图像', u'聚类图像']
# images = [img, dst]
# for i in range(2):
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
plt.figure(1)
plt.imshow(img,'gray')
plt.title('原始图像')
plt.xticks(()),plt.yticks(())
plt.figure(2)
plt.imshow(dst,'gray')
plt.title('聚类图像')
plt.xticks(()),plt.yticks(())
plt.show()
