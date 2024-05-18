import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')

#彩色图像转换成2维，每个像素点是一个包含三个值的向量，r，g，b三个值
data = img.reshape((-1,3))
data = np.float32(data)

#设置聚类参数
k = 8
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,10,1.0)
flag = cv2.KMEANS_RANDOM_CENTERS
attempts = 10

#k-means
ret,labels,centers = cv2.kmeans(data,k,None,criteria,attempts,flag)

#转换成像素值 0-255
centers = np.uint8(centers)
'''
#将每个像素的标签替换为对应的聚类中心颜色。
labels2.flatten() 将标签数组展平为一维数组。
centers2[labels2.flatten()] 通过标签索引，获取每个像素的聚类中心颜色
'''
res = centers[labels.flatten()]
dst = res.reshape((img.shape))

#转换彩色图
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
plt.imshow(dst)
plt.title("k-means")
plt.xticks([])
plt.yticks([])
plt.show()