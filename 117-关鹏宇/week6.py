# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def kmean(img, k):
    data = img.reshape((-1,3))
    data = np.float32(data)

    # 停止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签  初始质心位置
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类 聚集成k类
    compactness, labels2, centers2 = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    centers2 = np.uint8(centers2)
    res = centers2[labels2.flatten()]
    dst_img = res.reshape((img.shape))

    # 图像转换为RGB显示
    result = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

    return result

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']



img = cv2.imread('../lenna.png')
dst2 = kmean(img, 2)
dst4 = kmean(img, 4)
dst8 = kmean(img, 8)
dst16 = kmean(img, 16)
dst64 = kmean(img, 64)

images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()



