import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png",0)
print(img.dtype)
h,w = img.shape[:]
print(h,w)
data = img.reshape((h*w,1))
print(data)
data = np.float32(data)
print(data)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# K-means聚类
photos,lables,centers = cv2.kmeans(data,4,None,criteria,10,flags)

# 生成最终图像
dst = lables.reshape((img.shape[0],img.shape[1]))
print(dst)

# 用来显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
title = [u"原始图像",u"聚类图像"]
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray'),
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()

