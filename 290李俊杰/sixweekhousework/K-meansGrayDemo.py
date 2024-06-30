
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
'''
【第6周作业】
 实现kemans（全)
# 第二种实现
'''
# 对图片的灰度图进行K-mean聚类
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("lenna.png",0)
# print(type(img))

#在OpenCV中，传入Kmeans()函数的data数据要是一维，要把二维图片转成一维数据

h,w=img.shape
# print(h,w)
data=img.reshape((h*w,1))
#  data表示聚类数据，最好是np.flloat32类型的N维点集
data=np.float32(data)
# print(data)
# K表示聚类类簇数,设定类簇数
K=4
# 设置迭代停止的模式选择
criteria=(cv2.TERM_CRITERIA_EPS,20,2.0)
# 设置初始中心
flag=cv2.KMEANS_PP_CENTERS

# retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria,
# #                                      attempts, flags[, centers])
# 对数据进行聚类
retval, bestLabels, centers=cv2.kmeans(data,K,None,criteria,10,flag)
# print(bestLabels)
# 将返回的数组转回二维进行图片成像
# print(bestLabels)
img_kmeans=bestLabels.reshape((h,w))
# print(type(img_kmeans[0,0]),img_kmeans[0,0])
# print(img_kmeans)
# pjz=255//4
img_new=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if img_kmeans[i,j]==0:
            img_new[i,j]=0
        elif img_kmeans[i,j]==1:
            img_new[i,j]=255//4*2
        elif img_kmeans[i,j]==2:
            img_new[i,j]=255//4*3
        elif img_kmeans[i,j]==3:
            img_new[i,j]=255
cv2.imshow("k-means img",img_new)
cv2.waitKey(0)
# plt.subplot(221)
# plt.imshow(img_kmeans,"gray")
# plt.show()

