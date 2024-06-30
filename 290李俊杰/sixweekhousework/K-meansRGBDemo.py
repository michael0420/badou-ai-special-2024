'''
【第6周作业】
 实现kemans（全)
# 第二种实现
# retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
#     data表示聚类数据，最好是np.flloat32类型的N维点集
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 对彩色图进行K-mean聚类
img=cv2.imread("lenna.png")
# 将三维度数组(高，宽，通道)转换成了二维数组(高*宽， 通道)
data=img.reshape((-1,3))
data=np.float32(data)
# print(data)
k=2
bestLabels=None
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,3.0)
attempts=10
flag=cv2.KMEANS_RANDOM_CENTERS
retval, bestLabels, centers=cv2.kmeans(data,k,bestLabels,criteria,attempts,flag)
# print(bestLabels)


#图像转换回uint8二维类型
centers = np.uint8(centers)
res = centers[bestLabels.flatten()]
img_kmeans=res.reshape(img.shape)
#图像转换为RGB显示
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_kmeans = cv2.cvtColor(img_kmeans, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img_kmeans)
plt.show()

