import cv2 as cv
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

img = cv.imread('lenna.png', 0)
h,w = img.shape[:]

#二维图像转成一维,并转成float32类型
data = img.reshape((h * w, 1))
data = np.float32(data)

#设置迭代停止的模式
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

retval, bestLabels, centers = cv.kmeans(data, 4, None, criteria, 10, cv.KMEANS_PP_CENTERS)

newImg = bestLabels.reshape((h, w))

#用来正常显示测试标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
imgaes = [img, newImg]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(imgaes[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])


img1 = cv.imread('lenna.png')
data1 = img1.reshape((-1,3))
data1 = np.float32(data1)
#不同聚类效果测试
retval, bestLabels, centers = cv.kmeans(data1, 2, None, criteria, 10, cv.KMEANS_PP_CENTERS)
centers2 = np.uint8(centers)
res = centers2[bestLabels.flatten()]#降成一维
dst2 = res.reshape((img1.shape))
cv.imshow("k=2", dst2)

retval, bestLabels, centers = cv.kmeans(data1, 4, None, criteria, 10, cv.KMEANS_PP_CENTERS)
centers2 = np.uint8(centers)
res = centers2[bestLabels.flatten()]#降成一维
dst3 = res.reshape((img1.shape))
cv.imshow("k=4", dst3)

retval, bestLabels, centers = cv.kmeans(data1, 8, None, criteria, 10, cv.KMEANS_PP_CENTERS)
centers2 = np.uint8(centers)
res = centers2[bestLabels.flatten()]#降成一维
dst4 = res.reshape((img1.shape))
cv.imshow("k=8", dst4)

retval, bestLabels, centers = cv.kmeans(data1, 16, None, criteria, 10, cv.KMEANS_PP_CENTERS)
centers2 = np.uint8(centers)
res = centers2[bestLabels.flatten()]#降成一维
dst5 = res.reshape((img1.shape))
cv.imshow("k=16", dst5)

retval, bestLabels, centers = cv.kmeans(data1, 64, None, criteria, 10, cv.KMEANS_PP_CENTERS)
centers2 = np.uint8(centers)
res = centers2[bestLabels.flatten()]#降成一维
dst6 = res.reshape((img1.shape))
cv.imshow("k=64", dst6)

plt.show()
cv.waitKey(0)