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

##以cv2.KMEANS_RANDOM_CENTERS为例，进行K-Means聚类
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度图
img = cv2.imread('lenna.png',0)

#获取图像高度、宽度
h,w = img.shape

#图像二维像素转换为一维，转为float类型
data = img.reshape((-1,1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS ##随机初始化

#K-Means聚类 聚集成3类
compactness, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, flags)

#生成最终图像
dst = centers[labels.flatten()]
dst = dst.reshape((img.shape))
dst = dst.astype(np.uint8) #转换为uint8类型
print(dst)

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
## 用cv2显示图像

cv2.imshow('original', img)
cv2.imshow('kmeans', dst)
cv2.waitKey(0)

## 使用源码方式
# %load kmeans.py
from math import sqrt
#计算欧式距离
def eucDistance(vec1,vec2):
    return sqrt(sum(pow(vec2-vec1,2)))

#初始聚类中心选择
def initCentroids(dataSet,k):
    numSamples,dim = dataSet.shape
    centroids = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroids[i,:] = dataSet[index,:]
    return centroids

#K-means聚类算法，迭代
def kmeanss(dataSet,k):
    numSamples = dataSet.shape[0]
    clusterAssement = np.mat(np.zeros((numSamples,2)))
    clusterChanged = True
    #  初始化聚类中心
    centroids = initCentroids(dataSet,k)
    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 找到哪个与哪个中心最近
            for j in range(k):
                distance = eucDistance(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist = distance
                    minIndex = j
              # 更新簇
            clusterAssement[i,:] = minIndex,minDist**2
            if clusterAssement[i,0]!=minIndex:
                clusterChanged = True
         # 坐标均值更新簇中心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssement[:0].A==j)[0]]
            centroids[j,:] = np.mean(pointsInCluster,axis=0)
    print('Congratulations,cluster complete!')
    return centroids,clusterAssement
