# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k, criteria, flags):
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst

#读取原始图像
img = cv2.imread('../lenna.png')
print(img.shape)

#图像二维像素转换为一维
data = img.reshape((-1, 3))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类
dst2 = kmeans(data, 2, criteria, flags)
dst4 = kmeans(data, 4, criteria, flags)
dst8 = kmeans(data, 8, criteria, flags)
dst16 = kmeans(data, 16, criteria, flags)
dst64 = kmeans(data, 64, criteria, flags)

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
   plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
